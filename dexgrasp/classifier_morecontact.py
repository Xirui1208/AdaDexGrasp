import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import traceback
import torchvision.transforms as T
import open3d as o3d

script_dir = os.path.dirname(os.path.abspath(__file__))
pointnet2_path = os.path.abspath(os.path.join(script_dir, '../../../UniDexGrasp/Pointnet2_PyTorch'))
if os.path.exists(pointnet2_path):
    sys.path.append(pointnet2_path)
else:
    pointnet2_path = "/media/isaac/cf0a748c-87f5-4f21-b341-3a660e53cd74/home/user/xirui/UniDexGrasp/Pointnet2_PyTorch"
    if os.path.exists(pointnet2_path):
        sys.path.append(pointnet2_path)
    else:
        print("Error: Cannot find Pointnet2_PyTorch path.", file=sys.stderr)
        sys.exit(1)
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

def _sample_points_helper(pc, num_points):
    """Helper function to sample points."""
    if pc.shape[0] == 0: return torch.zeros(num_points, pc.shape[1], device=pc.device, dtype=pc.dtype)
    if pc.shape[0] > num_points:
        indices = torch.randperm(pc.shape[0])[:num_points]
        pc = pc[indices]
    elif pc.shape[0] < num_points:
        indices = torch.randint(0, pc.shape[0], (num_points - pc.shape[0],), device=pc.device)
        pc_append = pc[indices]
        pc = torch.cat([pc, pc_append], dim=0)
    return pc

def _augment_data_helper(xyz, normals):
    """Helper function for basic data augmentation."""
    theta = torch.rand(1, device=xyz.device) * 2 * np.pi
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rot_z_2x2 = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], device=xyz.device, dtype=xyz.dtype).squeeze()
    xyz[:, :2] = xyz[:, :2] @ rot_z_2x2.T
    if normals is not None:
        normals[:, :2] = normals[:, :2] @ rot_z_2x2.T

    scale = (torch.rand(1, device=xyz.device) * 0.2 + 0.9).item() # 0.9 ~ 1.1
    xyz = xyz * scale

    xyz += torch.randn_like(xyz) * 0.01
    return xyz, normals

def process_point_cloud(
    raw_pc,
    augment=False,
    num_points_hand=1024,
    num_points_object=1024,
    rgb_mean=None,
    rgb_std=None,
    rgb_scale_255=False,
    color_jitter_transform=None
):
    seg_id_hand = 2.0
    seg_id_object = 3.0
    seg_ids_contact_list = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    seg_ids_contact_tensor = torch.tensor(seg_ids_contact_list, device=raw_pc.device)
    num_contact_categories = len(seg_ids_contact_list)
    seg_ids_for_norm = raw_pc[:, 6]
    object_mask = (seg_ids_for_norm == seg_id_object)
    contact_mask = torch.isin(seg_ids_for_norm, seg_ids_contact_tensor)
    object_and_contact_mask = object_mask | contact_mask
    object_points = raw_pc[object_and_contact_mask, :3]
    centroid = torch.mean(object_points, dim=0) if object_points.shape[0] > 0 else torch.mean(raw_pc[:, :3], dim=0)
    all_xyz = raw_pc[:, :3] - centroid
    max_dist = torch.max(torch.sqrt(torch.sum((object_points - centroid)**2, dim=1))) if object_points.shape[0] > 0 else torch.max(torch.sqrt(torch.sum(all_xyz**2, dim=1)))
    if max_dist > 1e-6: all_xyz = all_xyz / max_dist
    raw_pc[:, :3] = all_xyz

    if augment:
        if color_jitter_transform is not None:
            rgb_normalized = raw_pc[:, 3:6] / 255.0 if rgb_scale_255 else raw_pc[:, 3:6]
            rgb_reshaped = rgb_normalized.T.unsqueeze(-1)
            if rgb_reshaped.shape[1] > 0:
                rgb_jittered = color_jitter_transform(rgb_reshaped).squeeze(-1).T
                raw_pc[:, 3:6] = rgb_jittered * 255.0 if rgb_scale_255 else rgb_jittered
        if raw_pc.shape[0] > num_points_hand:
            dropout_ratio = 0.2
            remaining_indices = torch.randperm(raw_pc.shape[0])[:int(raw_pc.shape[0] * (1 - dropout_ratio))]
            raw_pc = raw_pc[remaining_indices]
    
    all_xyz = raw_pc[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_xyz.cpu().numpy())
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    normals = torch.from_numpy(np.asarray(pcd.normals)).float().to(raw_pc.device)

    if augment:
        all_xyz, normals = _augment_data_helper(all_xyz, normals)

    raw_pc_with_normals = torch.cat([all_xyz, raw_pc[:, 3:6], normals, raw_pc[:, 6:]], dim=1)
    
    seg_ids = raw_pc_with_normals[:, 9] 
    pc_hand = raw_pc_with_normals[seg_ids == seg_id_hand]
    pc_object = raw_pc_with_normals[seg_ids == seg_id_object]
    contact_mask_split = torch.isin(seg_ids, seg_ids_contact_tensor)
    pc_contact = raw_pc_with_normals[contact_mask_split]

    is_contact_obj_features = torch.zeros(pc_object.shape[0], num_contact_categories, device=raw_pc.device)
    pc_object_w_feature = torch.cat([pc_object, is_contact_obj_features], dim=1)

    if pc_contact.shape[0] > 0:
        contact_seg_ids = pc_contact[:, 9]
        contact_seg_ids_zero_indexed = (contact_seg_ids - seg_ids_contact_list[0]).long()
        is_contact_con_features = F.one_hot(contact_seg_ids_zero_indexed, num_classes=num_contact_categories).float()
        
        pc_contact_w_feature = torch.cat([pc_contact, is_contact_con_features], dim=1)
    else:
        pc_contact_w_feature = torch.empty(0, pc_object_w_feature.shape[1], device=raw_pc.device)

    pc_object_full = torch.cat([pc_object_w_feature, pc_contact_w_feature], dim=0)

    if rgb_mean is not None and rgb_std is not None:
        if pc_hand.shape[0] > 0: 
            pc_hand[:, 3:6] = (pc_hand[:, 3:6] / 255.0 - rgb_mean) / rgb_std if rgb_scale_255 else (pc_hand[:, 3:6] - rgb_mean) / rgb_std
        if pc_object_full.shape[0] > 0: 
            pc_object_full[:, 3:6] = (pc_object_full[:, 3:6] / 255.0 - rgb_mean) / rgb_std if rgb_scale_255 else (pc_object_full[:, 3:6] - rgb_mean) / rgb_std

    pc_hand_sampled = _sample_points_helper(pc_hand, num_points_hand)
    pc_object_sampled = _sample_points_helper(pc_object_full, num_points_object)

    final_hand_pc = pc_hand_sampled[:, :9] 
    final_object_pc = torch.cat([pc_object_sampled[:, :9], pc_object_sampled[:, 10:]], dim=1)
    return final_hand_pc, final_object_pc


class _PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points = self.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = self.sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))
        new_points = torch.max(new_points, 2)[0].permute(0, 2, 1)
        return new_xyz, new_points

    def sample_and_group(self, npoint, radius, nsample, xyz, points):
        B, N, C = xyz.shape
        S = npoint
        if pointnet2_utils is None: raise RuntimeError("pointnet2_utils is not loaded.")
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
        idx = pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz)
        grouped_xyz = pointnet2_utils.grouping_operation(xyz.transpose(1, 2).contiguous(), idx).permute(0, 2, 3, 1)
        if points is not None:
            grouped_points = pointnet2_utils.grouping_operation(points.transpose(1, 2).contiguous(), idx).permute(0, 2, 3, 1)
            new_points = torch.cat([grouped_xyz - new_xyz.view(B, S, 1, C), grouped_points], dim=-1)
        else:
            new_points = grouped_xyz - new_xyz.view(B, S, 1, C)
        return new_xyz, new_points

    def sample_and_group_all(self, xyz, points):
        B, N, C = xyz.shape
        new_xyz = torch.zeros(B, 1, C, device=xyz.device)
        grouped_xyz = xyz.view(B, 1, N, C)
        if points is not None:
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        else:
            new_points = grouped_xyz
        return new_xyz, new_points

class PointNetEncoderBase(nn.Module):
    def __init__(self, input_feature_dim=3):
        super().__init__()
        # sa1: B, N, C+3 -> B, 256, 128
        self.sa1 = _PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=3 + input_feature_dim, mlp=[64, 64, 128], group_all=False)
        # sa2: B, 256, 128 -> B, 128, 256
        self.sa2 = _PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.output_dim = 256

    def forward(self, xyz_and_features):
        B, N, C = xyz_and_features.shape
        points_xyz = xyz_and_features[:, :, :3].contiguous()
        points_features = xyz_and_features[:, :, 3:].contiguous() if C > 3 else None
        
        if points_features is not None and points_features.shape[1] == 0:
            points_features = None
            
        l1_xyz, l1_points = self.sa1(points_xyz, points_features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        return l2_xyz, l2_points # B, 128, 256

class PointNetEncoderTop(nn.Module):
    def __init__(self, input_feature_dim=256, output_feature_dim=256, dropout=0.5):
        super().__init__()
        # sa3: B, 128, input_feature_dim -> B, 1, 1024
        self.sa3 = _PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=input_feature_dim + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, output_feature_dim)
        self.bn2 = nn.BatchNorm1d(output_feature_dim)
        self.dp2 = nn.Dropout(dropout)
        self.output_dim = output_feature_dim

    def forward(self, xyz, points):
        B, N, C = points.shape
        l3_xyz, l3_points = self.sa3(xyz, points) # l3_points: B, 1, 1024
        x = l3_points.view(B, 1024)
        x = self.dp1(F.relu(self.bn1(self.fc1(x))))
        x = self.dp2(F.relu(self.bn2(self.fc2(x))))
        return x

class GraspSuccessClassifier(nn.Module):
    def __init__(self, dropout=0.5, feature_dim=256, nhead=4, attention_dim=256):
        super().__init__()
        if pointnet2_utils is None: raise ImportError("pointnet2_ops is required.")

        self.hand_encoder_base = PointNetEncoderBase(input_feature_dim=6)
        self.object_encoder_base = PointNetEncoderBase(input_feature_dim=12) 
        
        base_output_dim = self.hand_encoder_base.output_dim
        if base_output_dim != attention_dim:
            attention_dim = base_output_dim
        self.hand_cross_attention = nn.MultiheadAttention(
            embed_dim=attention_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.object_cross_attention = nn.MultiheadAttention(
            embed_dim=attention_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )

        top_input_dim = attention_dim * 2 
        self.hand_encoder_top = PointNetEncoderTop(
            input_feature_dim=top_input_dim, output_feature_dim=feature_dim, dropout=dropout
        )
        self.object_encoder_top = PointNetEncoderTop(
            input_feature_dim=top_input_dim, output_feature_dim=feature_dim, dropout=dropout
        )

        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, hand_pc, object_pc):
        hand_xyz, hand_points = self.hand_encoder_base(hand_pc)
        object_xyz, object_points = self.object_encoder_base(object_pc)
        hand_attended, _ = self.hand_cross_attention(
            query=hand_points, key=object_points, value=object_points
        )
        object_attended, _ = self.object_cross_attention(
            query=object_points, key=hand_points, value=hand_points
        )
        hand_fused_points = torch.cat([hand_points, hand_attended], dim=-1)
        object_fused_points = torch.cat([object_points, object_attended], dim=-1)
        hand_features = self.hand_encoder_top(hand_xyz, hand_fused_points)
        object_features = self.object_encoder_top(object_xyz, object_fused_points)
        fused_features = torch.cat([hand_features, object_features], dim=1)
        return self.classification_head(fused_features)

class GraspDataset(Dataset):
    def __init__(self, data_path, num_points_hand=1024, num_points_object=1024, augment=False):
        self.augment = augment
        self.num_points_hand = num_points_hand
        self.num_points_object = num_points_object
        
        dataset = np.load(data_path)
        self.point_clouds = torch.from_numpy(dataset['point_clouds']).float()
        self.labels = torch.from_numpy(dataset['labels']).float()
        print(f"Dataset loaded from {data_path}")
        print(f"Raw point cloud shape: {self.point_clouds.shape}")
        label_counts = np.bincount(self.labels.numpy().astype(int))
        if len(label_counts) > 1: print(f"Label distribution: {label_counts[0]} samples for class 0, {label_counts[1]} samples for class 1. Augment={augment}")

        rgb = self.point_clouds[:, :, 3:6]
        self.rgb_scale_255 = (rgb.max() > 1.0).item()
        rgb01 = rgb / (255.0 if self.rgb_scale_255 else 1.0)
        self.rgb_mean = rgb01.mean(dim=(0, 1))
        self.rgb_std = rgb01.std(dim=(0, 1)).clamp_min(1e-6)

        if self.augment:
            self.color_jitter = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            ])
        else:
            self.color_jitter = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_pc = self.point_clouds[idx].clone()
        label = self.labels[idx]

        final_hand_pc, final_object_pc = process_point_cloud(
            raw_pc=raw_pc,
            augment=self.augment,
            num_points_hand=self.num_points_hand,
            num_points_object=self.num_points_object,
            rgb_mean=self.rgb_mean,
            rgb_std=self.rgb_std,
            rgb_scale_255=self.rgb_scale_255,
            color_jitter_transform=self.color_jitter
        )
        
        return final_hand_pc, final_object_pc, label

def train(model, train_loader, optimizer, criterion, device, label_smoothing=0.1, grad_clip_norm=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for hand_pc, object_pc, labels in train_loader:
        hand_pc, object_pc, labels = hand_pc.to(device), object_pc.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(hand_pc, object_pc)

        with torch.no_grad():
            smooth_labels = labels * (1.0 - label_smoothing) + label_smoothing / 2.0
        
        loss = criterion(logits, smooth_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return total_loss / max(len(train_loader), 1), 100.0 * correct / max(total, 1)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for hand_pc, object_pc, labels in test_loader:
            hand_pc, object_pc, labels = hand_pc.to(device), object_pc.to(device), labels.to(device).unsqueeze(1)
            logits = model(hand_pc, object_pc)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return total_loss / max(len(test_loader), 1), 100.0 * correct / max(total, 1)

if __name__ == '__main__':
    DATA_PATH = "classifier.npz"
    EPOCHS = 200
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 5e-4
    DROPOUT_RATE = 0.3
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP_NORM = 1.0
    FEATURE_DIM = 256
    ATTENTION_DIM = 256
    ATTENTION_HEADS = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_full_dataset = GraspDataset(data_path=DATA_PATH, augment=True)
    test_full_dataset = GraspDataset(data_path=DATA_PATH, augment=False)

    stats_path = "rgb_stats.npz"
    np.savez(stats_path, 
                rgb_mean=train_full_dataset.rgb_mean.cpu().numpy(),
                rgb_std=train_full_dataset.rgb_std.cpu().numpy(),
                rgb_scale_255=np.array(train_full_dataset.rgb_scale_255))
    print(f"RGB data saved to {stats_path}")
    
    num_pos = int((train_full_dataset.labels == 1).sum().item())
    num_neg = int((train_full_dataset.labels == 0).sum().item())
    pos_weight = torch.tensor(float(num_neg) / max(num_pos, 1e-6), device=device)
    print(f"pos_weight = {pos_weight.item():.4f} (pos={num_pos}, neg={num_neg})")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(sss.split(np.zeros(len(train_full_dataset.labels)), train_full_dataset.labels))
    
    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = GraspSuccessClassifier(
        dropout=DROPOUT_RATE, 
        feature_dim=FEATURE_DIM,
        nhead=ATTENTION_HEADS,
        attention_dim=ATTENTION_DIM
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
            
    print("\n--- Starting Training ---")
    best_test_acc = 0.0
    best_epoch = 0
    best_model_path = "best_classifier_model.pth" 
    patience = 20
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, LABEL_SMOOTHING, GRAD_CLIP_NORM)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        lr_before_step = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"Epoch [{epoch+1:03d}/{EPOCHS:03d}] | "
                f"LR: {lr_before_step:.1e} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%", end="")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"  <-- New best model saved to {best_model_path}!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print("")  

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break
    
    print(f"\n--- Training Finished ---")
    print(f"Best test accuracy achieved: {best_test_acc:.2f}% at epoch {best_epoch}")
    print(f"Best model saved to: {best_model_path}")
