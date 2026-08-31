import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type
from pointnet.pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNet2SemSegSSG(nn.Module):
    """
    Input: (B, C_in, N) 
    Output: (B, feat_dim, N)
    """
    def __init__(self, point_cloud_dim, feat_dim=128):
        super(PointNet2SemSegSSG, self).__init__()
        input_feature_dim = point_cloud_dim - 3 # C - 3
        
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=1024,
                radius=0.1,
                nsample=32,
                in_channel=3,
                mlp=[32, 32, 64],
                group_all=False,
            )
        )
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=256,
                radius=0.2,
                nsample=32,
                in_channel=64 + 3,
                mlp=[64, 64, 128],
                group_all=False,
            )
        )
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=64,
                radius=0.4,
                nsample=32,
                in_channel=128 + 3,
                mlp=[128, 128, 256],
                group_all=False,
            )
        )
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=16,
                radius=0.8,
                nsample=32,
                in_channel=256 + 3,
                mlp=[256, 256, 512],
                group_all=False,
            )
        )

        self.FP_modules_original = nn.ModuleList()
        self.FP_modules_original.append(
            PointNetFeaturePropagation(
                in_channel=128 + input_feature_dim, 
                mlp=[128, 128, 128]
            )
        )
        self.FP_modules_original.append(
            PointNetFeaturePropagation(in_channel=256 + 64, mlp=[256, 128])
        )
        self.FP_modules_original.append(
            PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        )
        self.FP_modules_original.append(
            PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        )
        
        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(True),
        )

    def forward(self, points):
        """
        Input:
            points: (B, C, N) tensor of input points (e.g., C=3 for xyz)
        Output:
            (B, feat_dim, N) tensor of per-point features
        """
        B, C, N = points.shape
        xyz = points[:, :3, :]  # xyz: B x 3 x N
        features = points[:, 3:, :] if C > 3 else None 
        l_xyz, l_features = [xyz], [features]
        for i, layer in enumerate(self.SA_modules):
            li_xyz, li_features = layer(l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules_original) + 1), -1):
            l_features[i - 1] = self.FP_modules_original[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class Network(nn.Module):
    def __init__(
        self, 
        point_cloud_dim: int,  
        num_classes: int,      
        feat_dim: int = 128    
    ):
        super(Network, self).__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.pointnet2 = PointNet2SemSegSSG(point_cloud_dim, feat_dim) 
        self.seg_head = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        )
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(self, pcs):
        if pcs.shape[1] == self.pointnet2.SA_modules[0].npoint:
             pcs_transposed = pcs.transpose(1, 2).contiguous()
        else:
            pcs_transposed = pcs.transpose(1, 2).contiguous()
        point_features = self.pointnet2(pcs_transposed)
        logits = self.seg_head(point_features)
        return logits


    def get_loss(self, pcs, labels):
        logits = self.forward(pcs)
        if labels.dtype != torch.long:
            labels = labels.long()
        total_loss = self.loss_fn(logits, labels)
        losses = {}
        losses["total_loss"] = total_loss
        return losses, logits