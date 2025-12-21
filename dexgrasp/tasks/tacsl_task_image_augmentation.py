import numpy as np
import torch
import torchvision.transforms as T
from tasks import image_transforms


class TacSLTaskImageAugmentation:
    """
    Class for performing image augmentation operations in the TacSL environment.
    Provides various methods for initializing, resetting, and applying image augmentations.
    """

    def init_image_augmentation(self):
        """
        Initialize image augmentation parameters and functions based on the configuration.
        """
        if self.cfg["randomize"]["randomize_color_channel"]:
            self.color_channel_order = np.tile([0, 1, 2], (self.num_envs, 1))

        if self.cfg["randomize"]["use_t_color_aug"]:
            t_color_aug_bcsh_level = self.cfg["randomize"]["t_color_aug_bcsh_level"]
            self.t_jitter_fn = T.ColorJitter(
                brightness=t_color_aug_bcsh_level[0],
                contrast=t_color_aug_bcsh_level[1],
                saturation=t_color_aug_bcsh_level[2],
                hue=t_color_aug_bcsh_level[3]
            )

        if self.cfg["randomize"]["use_ep_image_aug"]:
            ep_color_aug_bcsh_level = self.cfg["randomize"]["ep_color_aug_bcsh_level"]
            ep_image_aug_crop_scale = self.cfg["randomize"]["ep_image_aug_crop_scale"]
            ep_image_aug_aspect_k = self.cfg["randomize"]["ep_image_aug_aspect_k"]
            self.ep_jitter_fns = {}
            self.ep_resize_crop_fns = {}
            for k in self.image_obs_keys:
                self.ep_jitter_fns[k] = image_transforms.ColorJitterStateful(
                    num_envs=self.num_envs,
                    device=self.device,
                    brightness=ep_color_aug_bcsh_level[0],
                    contrast=ep_color_aug_bcsh_level[1],
                    saturation=ep_color_aug_bcsh_level[2],
                    hue=ep_color_aug_bcsh_level[3]
                )

                img_shape = self.obs_dims[k][:2]
                aspect = img_shape[1] * 1. / img_shape[0]
                self.ep_resize_crop_fns[k] = [image_transforms.RandomResizedCropStateful(
                    img_shape[:2],
                    scale=(ep_image_aug_crop_scale[0], ep_image_aug_crop_scale[1]),
                    ratio=(ep_image_aug_aspect_k[0] * aspect, ep_image_aug_aspect_k[1] * aspect)
                ) for _ in range(self.num_envs)]

        self.tactile_ig_keys = ['palm_tactile_camera_taxim', 'thumb_tactile_camera_taxim', 'ff_tactile_camera_taxim', 'mf_tactile_camera_taxim', 'rf_tactile_camera_taxim', 'lf_tactile_camera_taxim']
        if self.cfg["randomize"]["use_diff_tactile_img"] or self.cfg["randomize"]["concat_tactile_plain"]:
            images = self.get_camera_image_tensors_dict()
            ssr = self.cfg["tacsl"]["tactile_subsample_ratio"]
            self.default_tactile_img = {k: images[k][0][::ssr, ::ssr].clone() for k in self.tactile_ig_keys}
            self.default_tactile_img_envs = {k: images[k][:, ::ssr, ::ssr] for k in self.tactile_ig_keys}
            # for k in self.tactile_ig_keys:
            #     # images[k] = images[k][:, ::ssr, ::ssr]
            #     import cv2
            #     cv2.imshow(k, images[k].cpu().numpy()[0, ..., :3])
            #     cv2.waitKey(1)

    def reset_image_augmentation(self):
        """
        Reset image augmentation parameters for a new episode.
        """
        if self.cfg["randomize"]["randomize_color_channel"]:
            np.apply_along_axis(lambda x: np.random.shuffle(x), 1, self.color_channel_order)

        if self.cfg["randomize"]["use_ep_image_aug"]:
            for k in self.ep_jitter_fns:
                self.ep_jitter_fns[k].sample_transform()
                for env_id in range(self.num_envs):
                    self.ep_resize_crop_fns[k][env_id].sample_transform(self.obs_dims[k][0], self.obs_dims[k][1])

    def apply_image_augmentation_to_obs_dict(self):
        """
        Apply image augmentation to the observation dictionary.
        """
        if self.cfg_task.env.use_camera_obs:
            # initialize nominal tactile image for each parallel env
            if self.cfg_task.randomize.use_diff_tactile_img or self.cfg_task.randomize.concat_tactile_plain:
                for k in self.default_tactile_img_envs:
                    self.default_tactile_img_envs[k][:] = self.default_tactile_img[k]

            for cam in self.image_obs_keys:
                if cam in self.cfg_task.env.obsDims:
                    if self.cfg_task.randomize.randomize_color_channel:
                        im_shape = self.obs_dict[cam].shape
                        self.obs_dict[cam][:] = self.obs_dict[cam][torch.arange(self.num_envs)[:, None, None, None],
                        torch.arange(im_shape[1])[None, :, None, None],
                        torch.arange(im_shape[2])[None, None, :, None],
                        self.color_channel_order[:, None, None, :]]

                        if self.cfg_task.randomize.use_diff_tactile_img or self.cfg_task.randomize.concat_tactile_plain:
                            self.default_tactile_img_envs[cam][:] = self.default_tactile_img_envs[cam][
                                torch.arange(self.num_envs)[:, None, None, None],
                                torch.arange(im_shape[1])[None, :, None, None],
                                torch.arange(im_shape[2])[None, None, :, None],
                                self.color_channel_order[:, None, None, :]]

                    if self.cfg_task.randomize.use_t_color_aug:
                        self.obs_dict[cam][..., :3] = self.t_jitter_fn(
                            self.obs_dict[cam][..., :3].permute(0, 3, 1, 2)
                        ).permute(0, 2, 3, 1)

                    if self.cfg_task.randomize.use_ep_image_aug:
                        self.obs_dict[cam][..., :3] = self.ep_jitter_fns[cam](
                            self.obs_dict[cam][..., :3].permute(0, 3, 1, 2)
                        ).permute(0, 2, 3, 1)

                        if self.cfg_task.randomize.use_diff_tactile_img or self.cfg_task.randomize.concat_tactile_plain:
                            if cam in self.default_tactile_img_envs:
                                self.default_tactile_img_envs[cam] = self.ep_jitter_fns[cam](
                                    self.default_tactile_img_envs[cam].permute(0, 3, 1, 2)
                                ).permute(0, 2, 3, 1)
                        for env_id in range(self.num_envs):
                            self.obs_dict[cam][env_id][..., :3] = self.ep_resize_crop_fns[cam][env_id](
                                self.obs_dict[cam][env_id:env_id+1][..., :3].permute(0, 3, 1, 2)
                            ).permute(0, 2, 3, 1)

                            if self.cfg_task.randomize.use_diff_tactile_img or self.cfg_task.randomize.concat_tactile_plain:
                                if cam in self.default_tactile_img_envs:
                                    self.default_tactile_img_envs[cam][env_id] = self.ep_resize_crop_fns[cam][env_id](
                                        self.default_tactile_img_envs[cam][env_id:env_id+1].permute(0, 3, 1, 2)
                                    ).permute(0, 2, 3, 1)

                    if self.cfg_task.randomize.use_diff_tactile_img:
                        if cam in self.default_tactile_img_envs:
                            self.obs_dict[cam] -= self.default_tactile_img_envs[cam]/255.

                    if self.cfg_task.randomize.concat_tactile_plain:
                        if cam in self.default_tactile_img_envs:
                            self.obs_dict[cam][..., 3:6] = self.default_tactile_img_envs[cam]/255.
