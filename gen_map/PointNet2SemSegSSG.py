import torch
import torch.nn as nn
from models.pointnet.pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)


class PointNet2SemSegSSG(nn.Module):
    def __init__(self, feat_dim=128, use_normals=True):
        super(PointNet2SemSegSSG, self).__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=1024,
                radius=0.1,
                nsample=32,
                in_channel=3 + (6 if use_normals else 3),
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
                in_channel=128 + (6 if use_normals else 3), mlp=[128, 128, 128]
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
        points = points.transpose(1, 2)

        B, C, N = points.shape
        features = points  # features: B x C x N
        xyz = points[:, :3, :]  # xyz: B x 3 x N

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
