import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNetWithInstanceInfo
from typing import List, Optional, Tuple
import torch.nn.functional as F


class PointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = getPointNet({
                'input_feature_dim': self.pc_dim,
                'feat_dim': self.feature_dim
            })

        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)
            
    
    def forward(self, input_pc):
        others = {}
        return self.backbone(input_pc), others



class TransPointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int = 6,
        feature_dim: int = 128,
        state_dim: int = 191 + 29,
        use_seg: bool = True,
    ):
        super().__init__()

        cfg = {}
        cfg["state_dim"] = 191 + 29
        cfg["feature_dim"] = feature_dim
        cfg["pc_dim"] = pc_dim
        cfg["output_dim"] = feature_dim
        if use_seg:
            cfg["mask_dim"] = 2
        else:
            cfg["mask_dim"] = 0

        self.transpn = getPointNetWithInstanceInfo(cfg)

    def forward(self, input_pc):
        others = {}
        input_pc["pc"] = torch.cat([input_pc["pc"], input_pc["mask"]], dim = -1)
        return self.transpn(input_pc), others


# add spatial softmax module for tactile image

class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in [1].

    Concretely, the spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.

    References:
        [1]: End-to-End Training of Deep Visuomotor Policies,
        https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize=False):
        """Constructor.

        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        softmax = F.softmax(x.reshape(-1, h * w), dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        xc, yc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # concatenate and reshape the result
        # to (B, C*2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)

# Tactile Encoder using the SpatialSoftargmax
class TactileEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.spatial_soft_argmax = SpatialSoftArgmax()

    def forward(self, x):
        features = self.convs(x)
        coords = self.spatial_soft_argmax(features)
        return coords




class ActorCritic(nn.Module):

    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False, use_pc = False, is_tactile_image = True):
        super(ActorCritic, self).__init__()

        self.asymmetric = asymmetric
        self.use_pc = False
        self.backbone_type = model_cfg['backbone_type']
        self.freeze_backbone = model_cfg["freeze_backbone"]
        self.use_tactile_image = False

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        self.num_obs = 288
        
        if self.use_pc:
            self.num_obs = 179 + 29 #(robot_state)
            if self.backbone_type == "pn":
                self.backbone = PointNetBackbone(pc_dim=8, feature_dim=128)
            elif self.backbone_type =="transpn":
                self.backbone = TransPointNetBackbone(pc_dim=6, feature_dim=128)
            else:
                print("no such backbone")
                exit(123)
            print(self.backbone)
            self.num_obs += 128

# tactile image encoder part
        tactile_feature_dim = 0
        if self.use_tactile_image:
        #     self.num_tactile_sensors = 6 # 贴片数量
        #     self.tactile_image_height = 320
        #     self.tactile_image_width = 240
        #     self.tactile_image_channels = 3
        #     self.ssr = 4
        #     self.tactile_image_flat_dim = self.num_tactile_sensors * self.tactile_image_height * self.tactile_image_width * self.tactile_image_channels // (self.ssr * self.ssr) # 6 * 80 * 60 * 3 = 86400
        # if self.use_tactile_force_field:
            self.num_tactile_sensors = 6 # 贴片数量
            self.tactile_rows = 30
            self.tactile_cols = 20
            self.tactile_image_channels = 3
            self.tactile_image_flat_dim = self.num_tactile_sensors * self.tactile_rows * self.tactile_cols * self.tactile_image_channels # 6 * 30 * 20 * 3 = 10800
            # 实例化触觉编码器
            self.tactile_encoder = TactileEncoder(input_channels=self.tactile_image_channels)
            # 每个图像输出128维特征 (64 filters * 2 coords)
            tactile_feature_dim = self.num_tactile_sensors * 128 # 6 * 128 = 768
            print(f"Tactile Encoder loaded. Feature dim: {tactile_feature_dim}")
            self.num_obs += tactile_feature_dim



        actor_layers = []
        critic_layers = []
            
        actor_layers.append(nn.Linear(self.num_obs, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers.append(nn.Linear(self.num_obs, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    def act(self, observations, states):
        # print("OBSERVATIONS", observations.shape)
        # print("TACTILE", self.use_tactile_image)
        if self.use_pc and not self.freeze_backbone and not self.use_tactile_image:
            if self.backbone_type =="transpn":
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                pc_with_mask = torch.cat([pc, mask], dim = -1)
                pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_pc and self.freeze_backbone and not self.use_tactile_image:
            with torch.no_grad():
                if self.backbone_type =="transpn":
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                    pc_with_mask = torch.cat([pc, mask], dim = -1)
                    pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif not self.use_tactile_image:
            actions_mean = self.actor(observations[:,:288])
            # actions_mean = self.actor(observations)
        elif self.use_tactile_image and not self.use_pc:
            tactile_images = observations[:,288:288+self.tactile_image_flat_dim]
            B = tactile_images.shape[0]
            n = self.num_tactile_sensors
            h = self.tactile_rows
            w = self.tactile_cols
            c = self.tactile_image_channels
            tactile_images = tactile_images.reshape(B*n, c, h, w)
            tactile_features = self.tactile_encoder(tactile_images) # (B*n, 128)
            tactile_features = tactile_features.reshape(B, n*128) # (B, 768)
            observations = torch.cat([observations[:,:288], tactile_features], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_tactile_image and self.use_pc and not self.freeze_backbone:
            tactile_images = observations[:,288+8192:288+8192+self.tactile_image_flat_dim]
            B = tactile_images.shape[0]
            n = self.num_tactile_sensors
            h = self.tactile_rows
            w = self.tactile_cols
            c = self.tactile_image_channels
            tactile_images = tactile_images.reshape(B*n, c, h, w)
            tactile_features = self.tactile_encoder(tactile_images) # (B*n, 128)
            tactile_features = tactile_features.reshape(B, n*128) # (B, 768)
            if self.backbone_type =="transpn":
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :179], observations[:, 195:224]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                pc_with_mask = torch.cat([pc, mask], dim = -1)
                pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :179], observations[:, 195:224], tactile_features, pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_tactile_image and self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                tactile_images = observations[:,288+8192:288+8192+self.tactile_image_flat_dim]
                B = tactile_images.shape[0]
                n = self.num_tactile_sensors
                h = self.tactile_rows
                w = self.tactile_cols
                c = self.tactile_image_channels
                tactile_images = tactile_images.reshape(B*n, c, h, w)
                tactile_features = self.tactile_encoder(tactile_images) # (B*n, 128)
                tactile_features = tactile_features.reshape(B, n*128) # (B, 768)
                if self.backbone_type =="transpn":
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :179], observations[:, 195:224]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                    pc_with_mask = torch.cat([pc, mask], dim = -1)
                    pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :179], observations[:, 195:224], tactile_features, pc_feature], dim=1)
            actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

    def act_inference(self, observations):
        if self.use_pc and not self.freeze_backbone and not self.use_tactile_image:
            if self.backbone_type =="transpn":
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                pc_with_mask = torch.cat([pc, mask], dim = -1)
                pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_pc and self.freeze_backbone and not self.use_tactile_image:
            with torch.no_grad():
                if self.backbone_type =="transpn":
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                    pc_with_mask = torch.cat([pc, mask], dim = -1)
                    pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif not self.use_tactile_image:
            actions_mean = self.actor(observations[:,:288])
        elif self.use_tactile_image and not self.use_pc:
            tactile_images = observations[:,288:288+self.tactile_image_flat_dim]
            B = tactile_images.shape[0]
            n = self.num_tactile_sensors
            h = self.tactile_rows
            w = self.tactile_cols
            c = self.tactile_image_channels
            tactile_images = tactile_images.reshape(B*n, c, h, w)
            tactile_features = self.tactile_encoder(tactile_images) # (B*n, 128)
            tactile_features = tactile_features.reshape(B, n*128) # (B, 768)
            observations = torch.cat([observations[:,:288], tactile_features], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_tactile_image and self.use_pc and not self.freeze_backbone:
            tactile_images = observations[:,288+8192:288+8192+self.tactile_image_flat_dim]
            B = tactile_images.shape[0]
            n = self.num_tactile_sensors
            h = self.tactile_rows
            w = self.tactile_cols
            c = self.tactile_image_channels
            tactile_images = tactile_images.reshape(B*n, c, h, w)
            tactile_features = self.tactile_encoder(tactile_images) # (B*n, 128)
            tactile_features = tactile_features.reshape(B, n*128) # (B, 768)
            if self.backbone_type =="transpn":
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :179], observations[:, 195:224]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                pc_with_mask = torch.cat([pc, mask], dim = -1)
                pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :179], observations[:, 195:224], tactile_features, pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_tactile_image and self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                tactile_images = observations[:,288+8192:288+8192+self.tactile_image_flat_dim]
                B = tactile_images.shape[0]
                n = self.num_tactile_sensors
                h = self.tactile_rows
                w = self.tactile_cols
                c = self.tactile_image_channels
                tactile_images = tactile_images.reshape(B*n, c, h, w)
                tactile_features = self.tactile_encoder(tactile_images) # (B*n, 128)
                tactile_features = tactile_features.reshape(B, n*128) # (B, 768)
                if self.backbone_type =="transpn":
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :179], observations[:, 195:224]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                    pc_with_mask = torch.cat([pc, mask], dim = -1)
                    pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :179], observations[:, 195:224], tactile_features, pc_feature], dim=1)
            actions_mean = self.actor(observations)
        return actions_mean.detach()
        
    def evaluate(self, observations, states, actions):
        if self.use_pc and not self.freeze_backbone and not self.use_tactile_image:
            if self.backbone_type =="transpn":
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                pc_with_mask = torch.cat([pc, mask], dim = -1)
                pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_pc and self.freeze_backbone and not self.use_tactile_image:
            with torch.no_grad():
                if self.backbone_type =="transpn":
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :191], observations[:, 207:236]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:].reshape(-1, 1024, 2)
                    pc_with_mask = torch.cat([pc, mask], dim = -1)
                    pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :191], observations[:, 207:236], pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif not self.use_tactile_image:
            actions_mean = self.actor(observations[:,:288])
        elif self.use_tactile_image and not self.use_pc:
            tactile_images = observations[:,288:288+self.tactile_image_flat_dim]
            B = tactile_images.shape[0]
            n = self.num_tactile_sensors
            h = self.tactile_rows
            w = self.tactile_cols
            c = self.tactile_image_channels
            tactile_images = tactile_images.reshape(B*n, c, h, w)
            tactile_features = self.tactile_encoder(tactile_images) # (B*n, 128)
            tactile_features = tactile_features.reshape(B, n*128) # (B, 768)
            observations = torch.cat([observations[:,:288], tactile_features], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_tactile_image and self.use_pc and not self.freeze_backbone:
            tactile_images = observations[:,288+8192:288+8192+self.tactile_image_flat_dim]
            B = tactile_images.shape[0]
            n = self.num_tactile_sensors
            h = self.tactile_rows
            w = self.tactile_cols
            c = self.tactile_image_channels
            tactile_images = tactile_images.reshape(B*n, c, h, w)
            tactile_features = self.tactile_encoder(tactile_images) # (B*n, 128)
            tactile_features = tactile_features.reshape(B, n*128) # (B, 768)
            if self.backbone_type =="transpn":
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                data = {"pc": pc, "state": torch.cat([observations[:, :179], observations[:, 195:224]], dim=1), "mask": mask}
                pc_feature = self.backbone(data)[0].reshape(-1, 128)
            else:
                pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                pc_with_mask = torch.cat([pc, mask], dim = -1)
                pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :179], observations[:, 195:224], tactile_features, pc_feature], dim=1)
            actions_mean = self.actor(observations)
        elif self.use_tactile_image and self.use_pc and self.freeze_backbone:
            with torch.no_grad():
                tactile_images = observations[:,288+8192:288+8192+self.tactile_image_flat_dim]
                B = tactile_images.shape[0]
                n = self.num_tactile_sensors
                h = self.tactile_rows
                w = self.tactile_cols
                c = self.tactile_image_channels
                tactile_images = tactile_images.reshape(B*n, c, h, w)
                tactile_features = self.tactile_encoder(tactile_images) # (B*n, 128)
                tactile_features = tactile_features.reshape(B, n*128) # (B, 768)
                if self.backbone_type =="transpn":
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                    data = {"pc": pc, "state": torch.cat([observations[:, :179], observations[:, 195:224]], dim=1), "mask": mask}
                    pc_feature = self.backbone(data)[0].reshape(-1, 128)
                else:
                    pc = observations[:, 288:288+6144].reshape(-1, 1024, 6)
                    mask = observations[:, 288+6144:288+6144+2048].reshape(-1, 1024, 2)
                    pc_with_mask = torch.cat([pc, mask], dim = -1)
                    pc_feature = self.backbone(pc_with_mask)[0].reshape(-1, 128)
            observations = torch.cat([observations[:, :179], observations[:, 195:224], tactile_features, pc_feature], dim=1)
            actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()


        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

