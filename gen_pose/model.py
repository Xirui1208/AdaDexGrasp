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
    return nn.Sequential(*modules)


class PointNet2Backbone(nn.Module):

    def __init__(self, point_cloud_dim, feat_dim=128):
        super(PointNet2Backbone, self).__init__()
        input_feature_dim = point_cloud_dim - 3 
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=1024,
                radius=0.1,
                nsample=32,
                in_channel=point_cloud_dim,
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

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointNetFeaturePropagation(
                in_channel=128 + input_feature_dim,
                mlp=[128, 128, 128]
            )
        )
        self.FP_modules.append(
            PointNetFeaturePropagation(in_channel=256 + 64, mlp=[256, 128])
        )
        self.FP_modules.append(
            PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256])
        )
        self.FP_modules.append(
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
            points: (B, C, N) tensor (C=4, [X,Y,Z,Mask])
        Output:
            (B, feat_dim, N) tensor of per-point features
        """
        B, C, N = points.shape
        xyz = points[:, :3, :]
        features = points[:, 3:, :] if C > 3 else None 
        l_xyz, l_features = [xyz], [features]
        
        for i, layer in enumerate(self.SA_modules):
            li_xyz, li_features = layer(l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class NetworkRegression(nn.Module):
    def __init__(
        self, 
        point_cloud_dim: int, 
        robot_state_dim: int, 
        feat_dim: int = 128 
    ):
        super(NetworkRegression, self).__init__()

        self.feat_dim = feat_dim
        self.robot_state_dim = robot_state_dim

        self.pointnet2 = PointNet2Backbone(point_cloud_dim, feat_dim) 
        self.regression_head = create_mlp(
            input_dim=feat_dim,
            output_dim=robot_state_dim,
            net_arch=[feat_dim, feat_dim // 2]
        )
        self.loss_fn = nn.MSELoss()


    def forward(self, pcs):
        pcs_transposed = pcs.transpose(1, 2).contiguous()
        point_features = self.pointnet2(pcs_transposed) # (B, feat_dim, N)
        global_feature = torch.max(point_features, dim=2)[0] # (B, feat_dim)
        predicted_state = self.regression_head(global_feature) # (B, robot_state_dim)
        return predicted_state


    def get_loss(self, pcs, target_robot_state):
        predicted_state = self.forward(pcs)
        if target_robot_state.dtype != torch.float:
            target_robot_state = target_robot_state.float()
        total_loss = self.loss_fn(predicted_state, target_robot_state)
        losses = {}
        losses["total_loss"] = total_loss
        return losses, predicted_state