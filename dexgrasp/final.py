from unittest import TextTestRunner
import xxlimited
from matplotlib.pyplot import axis
import importlib.util
import hydra
import numpy as np
import os
import random
import yaml
import transforms3d
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from pyparsing import And
import torch
import cv2
import os.path as osp
from utils.torch_jit_utils import *
from utils.data_info import plane2euler
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from tasks.hand_base.base_task import BaseTask
from tasks.tacsl import TacSLSensors
from tasks.tacsl_task_image_augmentation import TacSLTaskImageAugmentation
from utils.visulize_contact import URDFHandCollisionModel, parse_collision_geometry_from_urdf
import pytorch3d.transforms
import atexit
import sys
sys.path.append(osp.realpath(osp.join(osp.realpath(__file__), '../../../..')))
from pointnet2_ops import pointnet2_utils
from diffusers import DDPMScheduler
from classifier_morecontact import GraspSuccessClassifier, process_point_cloud
import os
import pathlib
import hydra
import open3d as o3d
from omegaconf import OmegaConf
current_dir = pathlib.Path(__file__).parent.resolve()
grasp_dir = current_dir.parent
module_name = "Diffusion_Policy_3D"
dp3_policy_dir = grasp_dir / module_name
model_1 = "gen_map"
model1_policy_dir = grasp_dir / model_1
model_2 = "gen_pose"
model2_policy_dir = grasp_dir / model_2
sys.path.append(str(dp3_policy_dir))
sys.path.append(str(model1_policy_dir))
sys.path.append(str(model2_policy_dir))

train_file_path = dp3_policy_dir / "DP3.py"
train_model1_file_path = model1_policy_dir / "model.py"
train_model2_file_path = model2_policy_dir / "model.py"

spec = importlib.util.spec_from_file_location("dp3_train", str(train_file_path))
spec1 = importlib.util.spec_from_file_location("gen_map", str(train_model1_file_path))
spec2 = importlib.util.spec_from_file_location("gen_pose", str(train_model2_file_path))

dp3_train = importlib.util.module_from_spec(spec)
gen_map = importlib.util.module_from_spec(spec1)
gen_pose = importlib.util.module_from_spec(spec2)
sys.modules["dp3_train"] = dp3_train
sys.modules["gen_map"] = gen_map
sys.modules["gen_pose"] = gen_pose
spec.loader.exec_module(dp3_train)
spec1.loader.exec_module(gen_map)
spec2.loader.exec_module(gen_pose)

DP3 = dp3_train.DP3    
gen_map = gen_map.Network
gen_pose = gen_pose.NetworkRegression

def mov(tensor, device):
    return torch.from_numpy(tensor.cpu().numpy()).to(device)

def depth_image_to_point_cloud_GPU_batch(
        camera_depth_tensor_batch, camera_rgb_tensor_batch,
        camera_seg_tensor_batch, camera_view_matrix_inv_batch,
        camera_proj_matrix_batch, u, v, width: float, height: float,
        depth_bar: float, device: torch.device,
        # z_p_bar: float = 3.0,
        # z_n_bar: float = 0.3,
):
    
    batch_num = camera_depth_tensor_batch.shape[0]

    depth_buffer_batch = mov(camera_depth_tensor_batch, device)
    rgb_buffer_batch = mov(camera_rgb_tensor_batch, device) / 255.0
    seg_buffer_batch = mov(camera_seg_tensor_batch, device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv_batch = camera_view_matrix_inv_batch

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection

    proj_batch = camera_proj_matrix_batch
    fu_batch = 2 / proj_batch[:, 0, 0]
    fv_batch = 2 / proj_batch[:, 1, 1]

    centerU = width / 2
    centerV = height / 2

    Z_batch = depth_buffer_batch

    Z_batch = torch.nan_to_num(Z_batch, posinf=1e10, neginf=-1e10)

    X_batch = -(u.view(1, u.shape[-2], u.shape[-1]) - centerU) / width * Z_batch * fu_batch.view(-1, 1, 1)
    Y_batch = (v.view(1, v.shape[-2], v.shape[-1]) - centerV) / height * Z_batch * fv_batch.view(-1, 1, 1)

    R_batch = rgb_buffer_batch[..., 0].view(batch_num, 1, -1)
    G_batch = rgb_buffer_batch[..., 1].view(batch_num, 1, -1)
    B_batch = rgb_buffer_batch[..., 2].view(batch_num, 1, -1)
    S_batch = seg_buffer_batch.view(batch_num, 1, -1)

    valid_depth_batch = Z_batch.view(batch_num, -1) > -depth_bar

    Z_batch = Z_batch.view(batch_num, 1, -1)
    X_batch = X_batch.view(batch_num, 1, -1)
    Y_batch = Y_batch.view(batch_num, 1, -1)
    O_batch = torch.ones((X_batch.shape), device=device)

    position_batch = torch.cat((X_batch, Y_batch, Z_batch, O_batch, R_batch, G_batch, B_batch, S_batch), dim=1)
    # (b, N, 8)
    position_batch = position_batch.permute(0, 2, 1)
    position_batch[..., 0:4] = position_batch[..., 0:4] @ vinv_batch

    points_batch = position_batch[..., [0, 1, 2, 4, 5, 6, 7]]
    valid_batch = valid_depth_batch  # * valid_z_p_batch * valid_z_n_batch

    return points_batch, valid_batch

def sample_points(points, sample_num, sample_method, device):
    if sample_method == 'random':
        raise NotImplementedError

    elif sample_method == "furthest_batch":
        idx = pointnet2_utils.furthest_point_sample(points[:, :, :3].contiguous(), sample_num).long()
        idx = idx.view(*idx.shape, 1).repeat_interleave(points.shape[-1], dim=2)
        sampled_points = torch.gather(points, dim=1, index=idx)

    elif sample_method == 'furthest':
        eff_points = points[points[:, 2] > 0.04]
        eff_points_xyz = eff_points.contiguous()
        if eff_points.shape[0] < sample_num:
            eff_points = points[:, 0:3].contiguous()
        sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points_xyz.reshape(1, *eff_points_xyz.shape),
                                                                  sample_num)
        sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
    else:
        assert False
    return sampled_points

class hand_env(BaseTask,TacSLTaskImageAugmentation,TacSLSensors):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # ----- tactile image config current not use ----- #
        self.ssr = self.cfg["tacsl"]["tactile_subsample_ratio"]
        self.tactile_image_height = 320
        self.tactile_image_width = 240
        self.tactile_image_channels = 3
        self.num_tactile_sensors = 6
        self.use_tactile_image = self.cfg["tacsl"]["use_tactile_image"]
        self.use_force_field = self.cfg["tacsl"]["use_shear_force"]
        # ---------------------------------------------- #
        self.has_sensor = True
        self.up_axis = 'z'
        self.fingertips = ["ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal"]
        self.num_fingertips = len(self.fingertips)  ##
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.cfg["env"]["numStates"] = 0
        self.cfg["env"]["numActions"] = 24
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.asset_path = dict()

        self.table_dims = gymapi.Vec3(1, 1, 0.6)
        self.segmentation_id = {
            'hand': 2,
            'object': 3,
            'goal': 4,
            'table': 1,
            'contact': 5
        }
        self.camera_depth_tensor_list = []
        self.camera_rgb_tensor_list = []
        self.camera_seg_tensor_list = []
        self.camera_vinv_mat_list = []
        self.camera_proj_mat_list = []
        self.camera_handles = []
        self.num_cameras = len(self.cfg['env']['vision']['camera']['eye'])
        self._cfg_camera_props()
        self._cfg_camera_pose()

        camera_u = torch.arange(0, self.camera_props.width)
        camera_v = torch.arange(0, self.camera_props.height)
        self.camera_v2, self.camera_u2 = torch.meshgrid(
            camera_v, camera_u, indexing='ij')

        self.num_envs = self.cfg['env']['numEnvs']
        self.env_origin = torch.zeros((self.num_envs, 3), dtype=torch.float)

        self.x_n_bar = self.cfg['env']['vision']['bar']['x_n']
        self.x_p_bar = self.cfg['env']['vision']['bar']['x_p']
        self.y_n_bar = self.cfg['env']['vision']['bar']['y_n']
        self.y_p_bar = self.cfg['env']['vision']['bar']['y_p']
        self.z_n_bar = self.cfg['env']['vision']['bar']['z_n']
        self.z_p_bar = self.cfg['env']['vision']['bar']['z_p']
        self.depth_bar = self.cfg['env']['vision']['bar']['depth']
        self.num_pc_downsample = self.cfg['env']['vision']['pointclouds']['numDownsample']
        self.num_pc_presample = self.cfg['env']['vision']['pointclouds']['numPresample']
        self.num_each_pt = self.cfg['env']['vision']['pointclouds']['numEachPoint']
        self.num_pc_flatten = self.num_pc_downsample * self.num_each_pt # 6144
        super().__init__(cfg=self.cfg, enable_camera_sensors=True)

        # Vision
        self.camera_u2 = to_torch(self.camera_u2, device=self.device)
        self.camera_v2 = to_torch(self.camera_v2, device=self.device)
        self.env_origin = to_torch(self.env_origin, device=self.device)

        self.contact_calculation_enabled = True
        print("--- Initializing Hand Collision Model for Contact Map ---")
        # Construct robust paths relative to this file's location
        current_dir = os.path.dirname(__file__)
        collision_urdf_path = os.path.abspath(os.path.join(current_dir, "..", "assets", "urdf", "shadow_hand_tacsl", "robots", "movable_shadowhand_collision.urdf"))
        mesh_base_path = os.path.abspath(os.path.join(current_dir, "..", "assets", "urdf", "shadow_hand_tacsl"))
        self.pytorch3d_transforms = pytorch3d.transforms
        self.collision_data = parse_collision_geometry_from_urdf(collision_urdf_path)
        self.collision_model = URDFHandCollisionModel(urdf_path=collision_urdf_path,
                                                              mesh_path=mesh_base_path,
                                                              device=self.device,
                                                              collision_data=self.collision_data)
        print("--- Hand Collision Model Initialized Successfully ---")
        self.tactile_link_names = [
        'robot0:thumb_gelsight',
        'robot0:ff_gelsight',
        'robot0:mf_gelsight',
        'robot0:rf_gelsight',
        'robot0:lf_gelsight',
        'robot0:palm_gelsight'
    ]

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        if self.has_sensor:
            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs * 1 + self.num_object_dofs * 1)  ##
            self.dof_force_tensor = self.dof_force_tensor[:, :22]
        contact_force_net = self.gym.acquire_net_contact_force_tensor(self.sim)  # shape = (num_envs * num_bodies, 3)
        pairwise_contact_force_tensor = self.gym.acquire_pairwise_contact_force_tensor(self.sim)  # shape = (num_envs * num_bodies * num_bodies, 3)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_pairwise_contact_force_tensor(self.sim)

        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.contact_force_net = gymtorch.wrap_tensor(contact_force_net).view(self.num_envs, self.num_bodies, 3)[..., 0:3]
        self.pairwise_contact_force = gymtorch.wrap_tensor(pairwise_contact_force_tensor).view(self.num_envs, self.num_bodies, self.num_bodies, 3)[..., 0:3]
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.step_counter = 0
        
        #------------tacsl module current not use--------------#
        if self.cfg["tacsl"]["use_isaac_gym_tactile"]:
            assert self.cfg["tacsl"]["use_gelsight"], "shear force currently works only with gelsight fingers"
            self.initialize_tactile_rgb_camera()

        if self.cfg["tacsl"]["use_shear_force"]:
            assert self.cfg["tacsl"]["use_gelsight"], "shear force currently works only with gelsight fingers"
            num_divs = [self.cfg["tacsl"]["num_shear_rows"], self.cfg["tacsl"]["num_shear_cols"]]
            self.initialize_penalty_based_tactile(num_divs=num_divs)
        self.image_obs_keys = ['palm_tactile_camera_taxim', 'thumb_tactile_camera_taxim', 'ff_tactile_camera_taxim', 'mf_tactile_camera_taxim', 'rf_tactile_camera_taxim', 'lf_tactile_camera_taxim']

        self.init_image_augmentation()
        #-------------------------------------------------------#

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        self.gym.prepare_sim(self.sim)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        object_scale_dict = self.cfg['env']['object_code_dict']
        self.object_code_list = list(object_scale_dict.keys())
        all_scales = set()
        for object_scales in object_scale_dict.values():
            for object_scale in object_scales:
                all_scales.add(object_scale)
        self.id2scale = []
        self.scale2id = {}
        for scale_id, scale in enumerate(all_scales):
            self.id2scale.append(scale)
            self.scale2id[scale] = scale_id

        self.object_scale_id_list = []
        for object_scales in object_scale_dict.values():
            object_scale_ids = [self.scale2id[object_scale] for object_scale in object_scales]
            self.object_scale_id_list.append(object_scale_ids)

        self.repose_z = self.cfg['env']['repose_z']

        self.grasp_data = {}
        assets_path = '../assets'
        dataset_root_path = osp.join(assets_path, 'datasetv4.1')
        
        for object_code in self.object_code_list:
            data_per_object = {}
            dataset_path = dataset_root_path + '/' + object_code
            data_num_list = os.listdir(dataset_path)
            for num in data_num_list:
                data_dict = dict(np.load(os.path.join(dataset_path, num), allow_pickle=True))
                qpos = data_dict['qpos'].item() #goal
                scale_inverse = data_dict['scale'].item()  # the inverse of the object's scale
                scale = round(1 / scale_inverse, 2)
                assert scale in [0.06, 0.08, 0.10, 0.12, 0.15]
                target_qpos = torch.tensor(list(qpos.values())[:22], dtype=torch.float, device=self.device)
                target_hand_rot_xyz = torch.tensor(list(qpos.values())[22:25], dtype=torch.float, device=self.device)  # 3
                target_hand_rot = quat_from_euler_xyz(target_hand_rot_xyz[0], target_hand_rot_xyz[1], target_hand_rot_xyz[2])  # 4
                target_hand_pos = torch.tensor(list(qpos.values())[25:28], dtype=torch.float, device=self.device)
                plane = data_dict['plane']  # plane parameters (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
                translation, euler = plane2euler(plane, axes='sxyz')  # object
                object_euler_xy = torch.tensor([euler[0], euler[1]], dtype=torch.float, device=self.device)
                object_init_z = torch.tensor([translation[2]], dtype=torch.float, device=self.device)

                if object_init_z > 0.05:
                    continue

                if scale in data_per_object:
                    data_per_object[scale]['target_qpos'].append(target_qpos)
                    data_per_object[scale]['target_hand_pos'].append(target_hand_pos)
                    data_per_object[scale]['target_hand_rot'].append(target_hand_rot)
                    data_per_object[scale]['object_euler_xy'].append(object_euler_xy)
                    data_per_object[scale]['object_init_z'].append(object_init_z)
                else:
                    data_per_object[scale] = {}
                    data_per_object[scale]['target_qpos'] = [target_qpos]
                    data_per_object[scale]['target_hand_pos'] = [target_hand_pos]
                    data_per_object[scale]['target_hand_rot'] = [target_hand_rot]
                    data_per_object[scale]['object_euler_xy'] = [object_euler_xy]
                    data_per_object[scale]['object_init_z'] = [object_init_z]
            self.grasp_data[object_code] = data_per_object

        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg['env']['random_time']
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets/urdf/shadow_hand_tacsl"
        shadow_hand_asset_file = "robots/movable_shadowhand_collision.urdf"
        self.asset_path['shadow_hand'] = osp.join(asset_root, shadow_hand_asset_file)
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        self.shadow_hand_asset = shadow_hand_asset

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        self.actuators_name = ["FFJ4", "FFJ3", "FFJ2",  "LFJ5", "LFJ4", "LFJ3", "LFJ2", 
                               "MFJ4", "MFJ3", "MFJ2", "RFJ4", "RFJ3", "RFJ2",
                                "THJ5", "THJ4", "THJ3", "THJ2", "THJ1"]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in self.actuators_name]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # RandomLoad
        num_object_bodies_list = []
        num_object_shapes_list = []

        scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
        }

        assets_path = '../assets'
        visual_feat_root = osp.join(assets_path, 'meshdatav3_pc_feat')
        object_scale_idx_pairs = []
        self.visual_feat_data = {}
        self.visual_feat_buf = torch.zeros((self.num_envs, 64))

        for object_id in range(len(self.object_code_list)):
            object_code = self.object_code_list[object_id]
            self.visual_feat_data[object_id] = {}
            for scale_id in self.object_scale_id_list[object_id]:
                scale = self.id2scale[scale_id]
                if scale in self.grasp_data[object_code]:
                    object_scale_idx_pairs.append([object_id, scale_id])
                else:
                    print(f'prior not found: {object_code}/{scale}')
                file_dir = osp.join(visual_feat_root, f'{object_code}/pc_feat_{scale2str[scale]}.npy')
                with open(file_dir, 'rb') as f:
                    feat = np.load(f)
                self.visual_feat_data[object_id][scale_id] = torch.tensor(feat, device=self.device)


        object_asset_dict = {}
        mesh_path = osp.join(assets_path, 'meshdatav3_scaled')

        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 500
            object_asset_options.fix_base_link = False
            object_asset_options.disable_gravity = False
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None

            for obj_id, scale_id in object_scale_idx_pairs:
                if obj_id == object_id:
                    scale_str = scale2str[self.id2scale[scale_id]]
                    scaled_object_asset_file = object_code + f"/coacd/coacd_{scale_str}.urdf"
                    print("Loading asset: ", scaled_object_asset_file)
                    self.asset_path["object"] = os.path.abspath(os.path.join(mesh_path, scaled_object_asset_file))
                    scaled_object_asset = self.gym.load_asset(self.sim, mesh_path, scaled_object_asset_file, object_asset_options)
                    if obj_id not in object_asset_dict:
                        object_asset_dict[object_id] = {}
                    object_asset_dict[object_id][scale_id] = scaled_object_asset
                    if object_asset is None:
                        object_asset = scaled_object_asset
                                    
            assert object_asset is not None

            num_object_bodies_list.append(self.gym.get_asset_rigid_body_count(object_asset))
            num_object_shapes_list.append(self.gym.get_asset_rigid_shape_count(object_asset))

            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)

            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)


        # create table asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z,gymapi.AssetOptions())

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.8)  # gymapi.Vec3(0.1, 0.1, 0.65)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        object_start_pose = gymapi.Transform()
        self.object_rise = 0.1
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6 + self.object_rise)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * self.table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        self.shadow_hands = []
        self.envs = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.table_indices = []

        # RandomLoad
        self.num_obj_per_env = self.cfg['env']['random_load']['num_obj_per_env']
        self.num_actors_per_env = 2 + self.num_obj_per_env * 2
        self.init_object_waiting_pose = []
        self.all_object_indices = []
        self.object_indices = []
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        self.actor_handles = {}

        body_names = {
            'wrist': 'wrist',
            'palm': 'palm',
            'thumb': 'thdistal',
            'index': 'ffdistal',
            'middle': 'mfdistal',
            'ring': 'rfdistal',
            'little': 'lfdistal'
        }
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(shadow_hand_asset, body_name)
        
        if self.has_sensor:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)

        self.object_id_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_scale_buf = {}

        self.object_scale_idx_pairs = to_torch(object_scale_idx_pairs, dtype=torch.int32, device=self.device)

        def sample_idx(total, num_envs, num_per_env):
            num_total = num_envs * num_per_env
            div = num_total // total
            res = num_total % total
            assert div >= 1
            idx = list(torch.arange(total)) * div + list(torch.arange(res))
            idx = torch.tensor(idx).view(num_envs, num_per_env)
            return idx

        self.idx_tensor = sample_idx(self.object_scale_idx_pairs.shape[0], self.num_envs, self.num_obj_per_env)

        self.obj_actors = []
        for i in range(self.num_envs):
            self.obj_actors.append([])

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            idx_this_env = self.idx_tensor[i]
            object_idx_this_env = [idx.item() for idx in self.object_scale_idx_pairs[idx_this_env][:, 0]]

            if self.aggregate_mode >= 1:
                num_object_bodies = 0
                num_object_shapes = 0
                for obj_idx in object_idx_this_env:
                    num_object_bodies += num_object_bodies_list[obj_idx]
                    num_object_shapes += num_object_shapes_list[obj_idx]
                max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * num_object_bodies + 1
                max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * num_object_shapes + 1
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            shadow_hand_actor = self._load_shadow_hand(env_ptr, i, shadow_hand_asset, shadow_hand_dof_props, shadow_hand_start_pose)
            self.actor_handles['hand'] = shadow_hand_actor

            self.hand_start_states.append(
                [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                 shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                 shadow_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            id = int(i / self.num_envs * len(self.object_code_list))
            object_code = self.object_code_list[id]
            available_scale = []
            for scale_id in self.object_scale_id_list[id]:
                scale = self.id2scale[scale_id]
                if scale in self.grasp_data[object_code]:
                    available_scale.append(scale)
                else:
                    print(f'prior not found: {object_code}/{scale}')
            scale = available_scale[i % len(available_scale)]
            scale_id = self.scale2id[scale]
            self.object_scale_buf[i] = scale
            self.object_id_buf[i] = id

            self.visual_feat_buf[i] = self.visual_feat_data[id][scale_id]

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset_dict[id][scale_id], object_start_pose, "object", i, 0, self.segmentation_id['object'])
            self.actor_handles['object'] = object_handle

            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1.0)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # set friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            # Vision
            self._load_cameras(env_ptr, i, self.camera_props, self.camera_eye_list, self.camera_lookat_list)
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.active = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_actor_id_env   = self.gym.find_actor_index(env_ptr, 'object', gymapi.DOMAIN_ENV)
        object_rb_names  = self.gym.get_actor_rigid_body_names(self.envs[0], self.object_actor_id_env)
        self.object_body_id_env   = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.object_actor_id_env,   object_rb_names[0],  gymapi.DOMAIN_ENV)
        self.shadow_body_names = self.gym.get_actor_rigid_body_names(env_ptr, shadow_hand_actor)
        self.shadow_body_ids_env = dict()
        for b_name in self.shadow_body_names:
            self.shadow_body_ids_env[b_name] = self.gym.find_actor_rigid_body_index(
                self.envs[0], shadow_hand_actor, b_name, gymapi.DOMAIN_ENV)
        # tacsl module #
        self.set_elastomer_compliance(self.cfg["tacsl"]["compliance_stiffness"], self.cfg["tacsl"]["compliant_damping"])
        self._create_sensors(shadow_hand_actor,object_rb_names,self.asset_path)

    def set_elastomer_compliance(self, compliance_stiffness, compliant_damping):
        elastomer_links = [
            'palm_gelsight',
            'thumb_gelsight',
            'ff_gelsight',
            'mf_gelsight',
            'rf_gelsight',
            'lf_gelsight'
        ]
        for link_name in elastomer_links:
            self.configure_compliant_dynamics(
                actor_handle=self.actor_handles['hand'],
                elastomer_link_name=link_name,
                compliance_stiffness=compliance_stiffness,
                compliant_damping=compliant_damping,
                use_acceleration_spring=False
            )

    def _load_shadow_hand(self, env_ptr, env_id, shadow_hand_asset, hand_dof_props, init_hand_actor_pose):
        hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, init_hand_actor_pose, "hand", env_id, -1, self.segmentation_id['hand'])
        if self.has_sensor:
            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)
        self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)
        hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        self.hand_indices.append(hand_idx)
        hand_rigid_body_index = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ]
        agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
        hand_color = self.cfg['env']['vision']['color']['hand']
        for n in agent_index[0]:
            for m in n:
                for o in hand_rigid_body_index[m]:
                    self.gym.set_rigid_body_color(env_ptr, hand_actor, o, gymapi.MESH_VISUAL, gymapi.Vec3(*hand_color))
        return hand_actor

    def _cfg_camera_props(self):
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 256
        self.camera_props.height = 256
        self.camera_props.enable_tensors = True
        return

    def _cfg_camera_pose(self):
        self.camera_eye_list = []
        self.camera_lookat_list = []
        camera_eye_list = self.cfg['env']['vision']['camera']['eye']
        camera_lookat_list = self.cfg['env']['vision']['camera']['lookat']
        table_centor = np.array([0.0, 0.0, self.table_dims.z])
        for i in range(self.num_cameras):
            camera_eye = np.array(camera_eye_list[i]) + table_centor
            camera_lookat = np.array(camera_lookat_list[i]) + table_centor
            self.camera_eye_list.append(gymapi.Vec3(*list(camera_eye)))
            self.camera_lookat_list.append(gymapi.Vec3(*list(camera_lookat)))
        return

    def _load_cameras(self, env_ptr, env_id, camera_props, camera_eye_list, camera_lookat_list):
        camera_handles = []
        depth_tensors = []
        rgb_tensors = []
        seg_tensors = []
        vinv_mats = []
        proj_mats = []

        origin = self.gym.get_env_origin(env_ptr)
        self.env_origin[env_id][0] = origin.x
        self.env_origin[env_id][1] = origin.y
        self.env_origin[env_id][2] = origin.z
        for i in range(self.num_cameras):
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

            camera_eye = camera_eye_list[i]
            camera_lookat = camera_lookat_list[i]
            self.gym.set_camera_location(camera_handle, env_ptr, camera_eye, camera_lookat)
            raw_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            depth_tensor = gymtorch.wrap_tensor(raw_depth_tensor)
            depth_tensors.append(depth_tensor)

            raw_rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
            rgb_tensor = gymtorch.wrap_tensor(raw_rgb_tensor)
            rgb_tensors.append(rgb_tensor)

            raw_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
            seg_tensor = gymtorch.wrap_tensor(raw_seg_tensor)
            seg_tensors.append(seg_tensor)

            vinv_mat = torch.inverse((to_torch(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle), device=self.device)))
            vinv_mats.append(vinv_mat)

            proj_mat = to_torch(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)
            proj_mats.append(proj_mat)

            camera_handles.append(camera_handle)

        self.camera_depth_tensor_list.append(depth_tensors)
        self.camera_rgb_tensor_list.append(rgb_tensors)
        self.camera_seg_tensor_list.append(seg_tensors)
        self.camera_vinv_mat_list.append(vinv_mats)
        self.camera_proj_mat_list.append(proj_mats)

        return

    def compute_force(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        if self.has_sensor:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_pairwise_contact_force_tensor(self.sim)
        self.get_unpose_quat()        
        self.step_counter += 1
        thumb_force = self.pairwise_contact_force[:, self.object_body_id_env, self.shadow_body_ids_env['thumb_gelsight']]
        ff_force = self.pairwise_contact_force[:, self.object_body_id_env, self.shadow_body_ids_env['ff_gelsight']]
        mf_force = self.pairwise_contact_force[:, self.object_body_id_env, self.shadow_body_ids_env['mf_gelsight']]
        rf_force = self.pairwise_contact_force[:, self.object_body_id_env, self.shadow_body_ids_env['rf_gelsight']]
        lf_force = self.pairwise_contact_force[:, self.object_body_id_env, self.shadow_body_ids_env['lf_gelsight']]
        palm_force = self.pairwise_contact_force[:, self.object_body_id_env, self.shadow_body_ids_env['palm_gelsight']]
        self.finger_palm_contact_forces = torch.stack([thumb_force, ff_force, mf_force, rf_force, lf_force,palm_force], dim=1)

    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                -self.z_theta,
            )
        return

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_pc(self, pc):
        if self.repose_z:
            num_pts = pc.shape[1]
            return quat_apply(self.unpose_z_theta_quat.view(-1, 1, 4).expand(-1, num_pts, 4), pc)
        return pc

    def collect_pointclouds(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        depth_tensor = torch.stack([torch.stack(i) for i in self.camera_depth_tensor_list])
        rgb_tensor = torch.stack([torch.stack(i) for i in self.camera_rgb_tensor_list])
        seg_tensor = torch.stack([torch.stack(i) for i in self.camera_seg_tensor_list])
        vinv_mat = torch.stack([torch.stack(i) for i in self.camera_vinv_mat_list])
        proj_matrix = torch.stack([torch.stack(i) for i in self.camera_proj_mat_list])

        point_list = []
        valid_list = []
        for i in range(self.num_cameras):
            # (num_envs, num_pts, 7) (num_envs, num_pts)
            point, valid = depth_image_to_point_cloud_GPU_batch(depth_tensor[:, i], rgb_tensor[:, i], seg_tensor[:, i],
                                                                vinv_mat[:, i], proj_matrix[:, i],
                                                                self.camera_u2, self.camera_v2, self.camera_props.width,
                                                                self.camera_props.height, self.depth_bar, self.device,
                                                                # self.z_p_bar, self.z_n_bar
                                                                )
            point_list.append(point)
            valid_list.append(valid)

        # (num_envs, 65536 * num_cameras, 7)
        points = torch.cat(point_list, dim=1)
        depth_mask = torch.cat(valid_list, dim=1)
        points[:, :, :3] -= self.env_origin.view(self.num_envs, 1, 3)

        x_mask = (points[:, :, 0] > self.x_n_bar) * (points[:, :, 0] < self.x_p_bar)
        y_mask = (points[:, :, 1] > self.y_n_bar) * (points[:, :, 1] < self.y_p_bar)
        z_mask = (points[:, :, 2] > self.z_n_bar) * (points[:, :, 2] < self.z_p_bar)
        valid = depth_mask * x_mask * y_mask * z_mask

        point_nums = valid.sum(dim=1)
        now = 0
        points_list = []
        valid_points = points[valid]

        for env_id, point_num in enumerate(point_nums):
            points_all = valid_points[now: now + point_num]
            random_ids = torch.randint(0, points_all.shape[0], (self.num_pc_presample,), device=self.device,
                                       dtype=torch.long)
            points_all_rnd = points_all[random_ids]
            points_list.append(points_all_rnd)
            now += point_num
        
        assert len(points_list) == self.num_envs, f'{self.num_envs - len(points_list)} envs have 0 point'
        # (num_envs, num_pc_presample)
        points_batch = torch.stack(points_list)

        num_sample_dict = self.cfg['env']['vision']['pointclouds']
        if 'numSample' not in num_sample_dict.keys():

            points_fps = sample_points(points_batch, sample_num=self.num_pc_downsample,
                                       sample_method='furthest_batch', device=self.device)
        else:
            num_sample_dict = num_sample_dict['numSample']
            assert num_sample_dict['hand'] + num_sample_dict['object'] + num_sample_dict[
                'goal'] == self.num_pc_downsample

            zeros = torch.zeros((self.num_envs, self.num_pc_presample), device=self.device).to(torch.long)
            idx = torch.arange(self.num_envs * self.num_pc_presample, device=self.device).view(self.num_envs, self.num_pc_presample).to(torch.long)
            hand_idx = torch.where(points_batch[:, :, 6] == self.segmentation_id['hand'], idx, zeros)
            hand_pc = points_batch.view(-1, 7)[hand_idx]
            object_idx = torch.where(points_batch[:, :, 6] == self.segmentation_id['object'], idx, zeros)
            object_pc = points_batch.view(-1, 7)[object_idx]
            goal_idx = torch.where(points_batch[:, :, 6] == self.segmentation_id['goal'], idx, zeros)
            goal_pc = points_batch.view(-1, 7)[goal_idx]
            hand_fps = sample_points(hand_pc, sample_num=num_sample_dict['hand'],
                                     sample_method='furthest_batch', device=self.device)
            object_fps = sample_points(object_pc, sample_num=num_sample_dict['object'],
                                       sample_method='furthest_batch', device=self.device)
            goal_fps = sample_points(goal_pc, sample_num=num_sample_dict['goal'],
                                     sample_method='furthest_batch', device=self.device)
            points_fps = torch.cat([hand_fps, object_fps, goal_fps], dim=1)
        
        if self.contact_calculation_enabled:
            global_translation = self.root_state_tensor[self.hand_indices, 0:3]
            global_quat = self.root_state_tensor[self.hand_indices, 3:7]
            global_quat_w = global_quat[:, 3:4]  # Shape: [num_envs, 1]
            global_quat_xyz = global_quat[:, 0:3]  # Shape: [num_envs, 3]
            global_quat = torch.cat([global_quat_w, global_quat_xyz], dim=1)  # Shape: [num_envs, 4]
            global_rot_axis_angle = self.pytorch3d_transforms.matrix_to_axis_angle(
                self.pytorch3d_transforms.quaternion_to_matrix(global_quat)
            )
            joint_angles = self.shadow_hand_dof_pos
            if joint_angles.shape[1] == self.collision_model.n_dofs:
                hand_pose_for_contact = torch.cat([global_translation, global_rot_axis_angle, joint_angles], dim=1)
                pc_xyz_for_contact = points_fps[:, :, :3]
                object_mask_fps = (points_fps[:, :, 6] == self.segmentation_id['object'])
                seg_buffer = points_fps[:, :, 6]
                per_link_distances,object_local_pc = self.collision_model.get_signed_distances(hand_pose_for_contact, pc_xyz_for_contact)
                contact_force_magnitudes = torch.norm(self.finger_palm_contact_forces, p=2, dim=-1) # (num_envs, 6)
                force_threshold = 0.01
                has_contact_force = contact_force_magnitudes > force_threshold 
                contact_ids_map = [
                    11.0,  
                    12.0,  
                    13.0,  
                    14.0,  
                    15.0,  
                    16.0   
                ]# thumb, ff, mf, rf, lf, palm

                for i, link_name in reversed(list(enumerate(self.tactile_link_names))):

                    current_contact_id = contact_ids_map[i]
                    link_dist = per_link_distances[link_name].abs()
                    masked_dist = torch.where(object_mask_fps, link_dist, torch.full_like(link_dist, float('inf')))
                    contact_env_indices = torch.where(has_contact_force[:, i])[0]
                    
                    if contact_env_indices.numel() > 0:
                        dists_for_contact_envs = masked_dist[contact_env_indices]
                        _, topk_indices = torch.topk(dists_for_contact_envs, 30, dim=1, largest=False)
                        seg_buffer[contact_env_indices.unsqueeze(1), topk_indices] = current_contact_id
                
                points_fps[:, :, 6] = seg_buffer
                
            else:
                print(f"!!! WARNING: DOF mismatch between sim ({joint_angles.shape[1]}) and collision model ({self.collision_model.n_dofs}). Skipping contact calculation.")

        mask_hand = (points_fps[:,:,6] == self.segmentation_id["hand"])
        mask_object = (points_fps[:,:,6] == self.segmentation_id["object"])
        
        mask_contact_thumb = (points_fps[:, :, 6] == 11.0)
        mask_contact_ff = (points_fps[:, :, 6] == 12.0)
        mask_contact_mf = (points_fps[:, :, 6] == 13.0)
        mask_contact_rf = (points_fps[:, :, 6] == 14.0)
        mask_contact_lf = (points_fps[:, :, 6] == 15.0)
        mask_contact_palm = (points_fps[:, :, 6] == 16.0)
        
        mask_contact = mask_contact_thumb | mask_contact_ff | mask_contact_mf | \
                       mask_contact_rf | mask_contact_lf | mask_contact_palm

        others = {}
        others["mask_hand"] = mask_hand
        others["mask_object"] = mask_object
        others["mask_contact"] = mask_contact      
        others["mask_contact_thumb"] = mask_contact_thumb
        others["mask_contact_ff"] = mask_contact_ff
        others["mask_contact_mf"] = mask_contact_mf
        others["mask_contact_rf"] = mask_contact_rf
        others["mask_contact_lf"] = mask_contact_lf
        others["mask_contact_palm"] = mask_contact_palm

        self.pc = points_fps.clone()

        if self.repose_z:
            pc_xyz = self.unpose_pc(points_fps[:, :, 0:3])
            points_fps = torch.cat([pc_xyz, points_fps[:, :, 3:]], dim=-1)
        else:
            points_fps = points_fps

        self.gym.end_access_image_tensors(self.sim)
     
    def refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_pairwise_contact_force_tensor(self.sim)

def set_sim_params(cfg):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.physx.solver_type = 1
    sim_params.use_gpu_pipeline = True
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.physx.num_subscenes = 4
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.physx.contact_collection = gymapi.ContactCollection(1)
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)
    return sim_params

def load_classifier_model(model_path, device):
    model = GraspSuccessClassifier(
        dropout=0.3, 
        feature_dim=256, 
        nhead=4, 
        attention_dim=256
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def _normalize_pc(pc):
    xyz = pc[:, :3]
    centroid = torch.mean(xyz, dim=0)
    xyz = xyz - centroid
    max_dist = torch.max(torch.sqrt(torch.sum(xyz**2, dim=1)))
    if max_dist > 1e-6:
        xyz = xyz / max_dist
    pc[:, :3] = xyz
    return pc
   
def prepare_classifier_input(full_pc, device=None, num_points_hand=512, num_points_object=512, 
                             rgb_mean=None, rgb_std=None, rgb_scale_255=False):

    if full_pc.dim() == 2:
        full_pc = full_pc.unsqueeze(0)
    assert full_pc.dim() == 3 and full_pc.size(-1) >= 7, "full_pc must be (B,N,7)"
    if device is None:
        device = full_pc.device
    else:
        device = torch.device(device)

    full_pc = full_pc.to(device=device, dtype=torch.float32)

    B = full_pc.shape[0]
    hand_list = []
    object_list = []
    for i in range(B):
        pc = full_pc[i]  # (N,7)

        hand_pc, object_pc = process_point_cloud(
            raw_pc=pc,
            augment=False,
            num_points_hand=num_points_hand,
            num_points_object=num_points_object,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            rgb_scale_255=rgb_scale_255,
            color_jitter_transform=None
        )
        hand_list.append(hand_pc)
        object_list.append(object_pc)

    hand_pc_batch = torch.stack(hand_list, dim=0)
    object_pc_batch = torch.stack(object_list, dim=0)

    info = {'batch_size': B}
    
    return hand_pc_batch, object_pc_batch, info

def run_classifier(model, point_cloud, robot_state=None, attempt_num=0, return_prob=False, 
                   evaluator_instance=None):   

    model_device = next(model.parameters()).device

    rgb_mean = evaluator_instance.rgb_mean.to(model_device)
    rgb_std = evaluator_instance.rgb_std.to(model_device)
    rgb_scale_255 = evaluator_instance.rgb_scale_255

    hand_pc, object_pc, info = prepare_classifier_input(point_cloud, device=model_device,
                                                        rgb_mean=rgb_mean, 
                                                        rgb_std=rgb_std, 
                                                        rgb_scale_255=rgb_scale_255)
    with torch.no_grad():
        hand_pc = hand_pc.to(model_device)
        object_pc = object_pc.to(model_device)
        logits = model(hand_pc, object_pc)
        probs = torch.sigmoid(logits.view(-1)).detach().cpu()
        is_success = (probs > 0.5).cpu()  # 0.5
    bools = is_success.numpy().tolist()
    prob_list = probs.numpy().tolist()
    
    if return_prob:
        return bools, prob_list, info
    if len(bools) == 1:
        return bools[0]
    return bools


class GraspEvaluator:

    def __init__(self, args):
        print("--- Initial ---")
        yaml_path = args.cfg_env
        self.cfg = yaml.safe_load(open(yaml_path, 'r'))
        self.sim_params = set_sim_params(self.cfg)
        self.task = hand_env(
            cfg=self.cfg,
            sim_params=self.sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            device_type='cuda',
            device_id=0,
            headless=args.headless
        )
        self.gym = self.task.gym
        self.sim = self.task.sim
        self.device = self.task.device
        stats_path = "rgb_stats.npz"       
        self.rgb_mean = torch.zeros(3, device=self.device)
        self.rgb_std = torch.ones(3, device=self.device)
        self.rgb_scale_255 = False
        stats = np.load(stats_path)
        self.rgb_mean = torch.from_numpy(stats['rgb_mean']).float().to(self.device)
        self.rgb_std = torch.from_numpy(stats['rgb_std']).float().to(self.device)
        self.rgb_scale_255 = bool(stats['rgb_scale_255'].item())
        print(f"RGB load success: Mean={self.rgb_mean.cpu().numpy()}, Std={self.rgb_std.cpu().numpy()}, Scale255={self.rgb_scale_255}")
        self.classifier_model = load_classifier_model(args.classifier_model_path, self.device)
        self.contact_map_model = gen_map(point_cloud_dim=3,num_classes=7,)
        self.contact_map_model.load_state_dict(
            torch.load(args.contact_map_model_path, map_location=self.device)
        )
        self.contact_map_model.to(self.device)
        self.contact_map_model.eval()
        self.contact_pose_model = gen_pose(point_cloud_dim=4,robot_state_dim=29)
        self.contact_pose_model.load_state_dict(
            torch.load(args.contact_pose_model_path, map_location=self.device)
        )
        self.contact_pose_model.to(self.device)
        self.contact_pose_model.eval()
        self.dp3_policy = DP3( checkpoint_path=args.diffusion_model_path, device=self.device )

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.task.viewer, self.task.sim, True)
        self.task.refresh_tensors()
        self.object_fly_threshold_z = 0.95
        atexit.register(self.cleanup)


    def set_hand_pose_from_cvae(self, env_ids):
        pc_obs, _ = self.get_force_pc()
        all_env_ids_succ = []
        all_root_pos = []
        all_root_rot = []
        all_dof_pos = []

        with torch.no_grad():
            for i in env_ids:
                env_id = i.item()
                pc_env_i = pc_obs[env_id]
                obj_seg_id = self.task.segmentation_id['object']
                object_mask = (pc_env_i[:, 6] == obj_seg_id)
                object_points_full = pc_env_i[object_mask]
                normalized_object_points_full = _normalize_pc(object_points_full.clone())
                object_points_xyz_normalized = normalized_object_points_full[:, :3]
                model_input_m1 = object_points_xyz_normalized.unsqueeze(0)
                predicted_logits = self.contact_map_model(model_input_m1)
                predicted_classes = torch.argmax(predicted_logits, dim=1).squeeze(0)
                model_input_m2 = torch.cat(
                    [object_points_xyz_normalized, predicted_classes.float().unsqueeze(-1)],
                    dim=1
                ).unsqueeze(0)
                generated_rel_state = self.contact_pose_model(model_input_m2)
                root_state_o = generated_rel_state[0, 0:7].unsqueeze(0)
                dof_state_22 = generated_rel_state[0, 7:29]
                object_state = self.task.root_state_tensor[self.task.object_indices[env_id]].unsqueeze(0)
                obj_pos_w = object_state[:, 0:3]
                obj_rot_w = object_state[:, 3:7]
                pos_o = root_state_o[:, 0:3]
                rot_o = root_state_o[:, 3:7]
                pos_w_corrected = quat_apply(obj_rot_w, pos_o) + obj_pos_w
                rot_w_corrected = quat_mul(obj_rot_w, rot_o)
                all_env_ids_succ.append(env_id)
                all_root_pos.append(pos_w_corrected.squeeze(0))
                all_root_rot.append(rot_w_corrected.squeeze(0))
                all_dof_pos.append(dof_state_22)
        env_ids_tensor = torch.tensor(all_env_ids_succ, dtype=torch.long, device=self.device)
        hand_indices_tensor = self.task.hand_indices[env_ids_tensor].to(torch.int32)
        root_pos_tensor = torch.stack(all_root_pos)
        root_rot_tensor = torch.stack(all_root_rot)
        dof_pos_tensor = torch.stack(all_dof_pos)
        hand_indices_long = hand_indices_tensor.long()
        self.task.root_state_tensor[hand_indices_long, 0:3] = root_pos_tensor
        self.task.root_state_tensor[hand_indices_long, 3:7] = root_rot_tensor
        self.task.root_state_tensor[hand_indices_long, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.task.root_state_tensor),
            gymtorch.unwrap_tensor(hand_indices_tensor),
            len(hand_indices_tensor)
        )
        self.task.shadow_hand_dof_pos[env_ids_tensor, :] = dof_pos_tensor
        self.task.shadow_hand_dof_vel[env_ids_tensor, :] = 0.0
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.task.dof_state),
            gymtorch.unwrap_tensor(hand_indices_tensor),
            len(hand_indices_tensor)
        )

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        for env_id in env_ids:
            i = env_id.item()
            object_id = self.task.object_id_buf[i].item()
            object_code = self.task.object_code_list[object_id]
            scale = self.task.object_scale_buf[i]
            grasp_options = self.task.grasp_data[object_code][scale]
            num_options = len(grasp_options['target_qpos'])
            chosen_idx = random.randint(0, num_options - 1)
            self.task.target_qpos[i:i + 1] = grasp_options['target_qpos'][chosen_idx]
            self.task.target_hand_pos[i:i + 1] = grasp_options['target_hand_pos'][chosen_idx]
            self.task.target_hand_rot[i:i + 1] = grasp_options['target_hand_rot'][chosen_idx]
            self.task.object_init_euler_xy[i:i + 1] = grasp_options['object_euler_xy'][chosen_idx]
            self.task.object_init_z[i:i + 1] = grasp_options['object_init_z'][chosen_idx]
            hand_indices = self.task.hand_indices[env_ids]
            self.task.root_state_tensor[hand_indices, :] = self.task.hand_start_states[env_ids, :]
            self.task.shadow_hand_dof_pos[env_ids, :] = self.task.shadow_hand_dof_default_pos
            self.task.shadow_hand_dof_vel[env_ids, :] = 0.0
            hand_indices_int32 = hand_indices.to(torch.int32)
        theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]
        new_object_rot = quat_from_euler_xyz(
            self.task.object_init_euler_xy[env_ids, 0],
            self.task.object_init_euler_xy[env_ids, 1],
            theta
        )
        num_resets = len(env_ids)
        new_pos_xy = torch.zeros((num_resets, 2), device=self.device, dtype=torch.float)
        table_surface_z = self.task.table_dims.z * 0.5
        new_pos_z = table_surface_z + self.task.object_init_z[env_ids]
        new_pos = torch.cat([new_pos_xy, new_pos_z], dim=1)
        self.task.root_state_tensor[self.task.object_indices[env_ids], 0:3] = new_pos
        self.task.root_state_tensor[self.task.object_indices[env_ids], 3:7] = new_object_rot
        object_indices_int32 = self.task.object_indices[env_ids].to(torch.int32)
        all_indices_to_reset = torch.cat([hand_indices_int32, object_indices_int32])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.task.root_state_tensor),
            gymtorch.unwrap_tensor(all_indices_to_reset),
            len(all_indices_to_reset)
        )
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim) 
        self.gym.draw_viewer(self.task.viewer, self.task.sim, True)
        self.task.refresh_tensors()
        self.set_hand_pose_from_cvae(env_ids)

    def get_force_pc(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_pairwise_contact_force_tensor(self.sim)
        self.task.compute_force() 
        self.task.collect_pointclouds()
        point_cloud = self.task.pc.clone()
        hand_root_state = self.task.root_state_tensor[self.task.hand_indices,0:7]
        robot_state = torch.cat([hand_root_state, self.task.shadow_hand_dof_pos], dim=1)
        return point_cloud, robot_state       

    def run_evaluation(self,max_adjustments=10):
        from near import normilize_batch
        num_envs = self.task.num_envs
        max_trials = max_adjustments + 1
        env_states = torch.zeros(num_envs, dtype=torch.int, device=self.device)
        total_successes_per_env = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        current_trial_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        classifier_predictions = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        initial_object_z = torch.zeros(num_envs, device=self.device)
        lift_step_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        hold_step_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        lift_height = 0.05
        lift_steps = 60
        hold_steps = 60
        lift_force_magnitude = 50.0 
        self.task.reset_buf.fill_(1)
        all_env_ids = torch.arange(num_envs, device=self.device)
        self.reset(all_env_ids) 
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.task.viewer, self.task.sim, True)
        self.task.refresh_tensors() 
        total_contact_force_per_env = torch.zeros(num_envs, device=self.device)
        hand_part_ids = [
            self.task.shadow_body_ids_env['thumb_gelsight'],
            self.task.shadow_body_ids_env['ff_gelsight'],
            self.task.shadow_body_ids_env['mf_gelsight'],
            self.task.shadow_body_ids_env['rf_gelsight'],
            self.task.shadow_body_ids_env['lf_gelsight'],
            self.task.shadow_body_ids_env['palm_gelsight'],
        ]
        object_body_id = self.task.object_body_id_env
        for part_id in hand_part_ids:
            force_vector = self.task.pairwise_contact_force[:, object_body_id, part_id]
            total_contact_force_per_env += torch.norm(force_vector, p=2, dim=-1)
        # 0-initial 1-lift 2-hold 3-prepare next trial 4-done
        while torch.any(env_states != 4):
            pending_envs_mask = (env_states == 0)
            pending_env_ids = pending_envs_mask.nonzero(as_tuple=False).squeeze(-1)
            if len(pending_env_ids) > 0:
                print("Evaluating pending envs:", pending_env_ids.cpu().numpy())
                pc_obs, state_obs = self.get_force_pc()
                batch_pc = pc_obs[pending_env_ids]
                is_success_list, probs, _ = run_classifier(self.classifier_model, batch_pc, 
                                                           return_prob=True, 
                                                           evaluator_instance=self)
                for i, env_id in enumerate(pending_env_ids):
                    env_idx = env_id.item() 
                    is_success = is_success_list[i]
                    if is_success:
                        env_states[env_idx] = 1 # -> 1 (Lifting)
                        initial_object_z[env_idx] = self.task.root_state_tensor[self.task.object_indices[env_idx], 2]
                        lift_step_counts[env_idx] = 0 
                        hold_step_counts[env_idx] = 0 
                    else:
                        env_states[env_idx] = 3 # -> 3 (Prepare for Adjustment)

            lifting_envs_mask = (env_states == 1)
            if torch.any(lifting_envs_mask):
                lifting_env_ids = lifting_envs_mask.nonzero(as_tuple=False).squeeze(-1)
                self.task.apply_forces[lifting_env_ids, self.task.shadow_body_ids_env['palm'], 2] = lift_force_magnitude
                lift_step_counts[lifting_env_ids] += 1

            self.gym.apply_rigid_body_force_tensors(
                self.sim, 
                gymtorch.unwrap_tensor(self.task.apply_forces), 
                gymtorch.unwrap_tensor(self.task.apply_torque), 
                gymapi.ENV_SPACE
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.task.viewer, self.task.sim, True)
            self.task.refresh_tensors()
            
            self.task.apply_forces.zero_()
            self.task.apply_torque.zero_()

            finished_lift_mask = (lift_step_counts >= lift_steps) & (env_states == 1)
            if torch.any(finished_lift_mask):
                finished_lift_ids = finished_lift_mask.nonzero(as_tuple=False).squeeze(-1)
                print(f"Env {finished_lift_ids.cpu().numpy()} finished lifting, now holding.")
                env_states[finished_lift_ids] = 2 #  -> 2 (Holding)

            holding_envs_mask = (env_states == 2)
            if torch.any(holding_envs_mask):
                hold_step_counts[holding_envs_mask] += 1

            finished_hold_mask = (hold_step_counts >= hold_steps) & (env_states == 2)
            if torch.any(finished_hold_mask):
                finished_hold_ids = finished_hold_mask.nonzero(as_tuple=False).squeeze(-1)
                current_object_z = self.task.root_state_tensor[self.task.object_indices[finished_hold_ids], 2]
                is_lifted = (current_object_z - initial_object_z[finished_hold_ids]) > (lift_height * 0.5)
                for i, env_id in enumerate(finished_hold_ids):
                    env_idx = env_id.item()
                    if is_lifted[i]:
                        print(f"Env {env_idx} grasp SUCCEEDED.")
                        total_successes_per_env[env_idx] = 1
                    else:
                        print(f"Env {env_idx} grasp FAILED.")
                    env_states[env_idx] = 4 # -> 4 (Done)
            
            prepare_mask = (env_states == 3)
            prepare_env_ids = prepare_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if len(prepare_env_ids) > 0:
                current_trial_counts[prepare_env_ids] += 1
                needs_next_trial_mask = (current_trial_counts < max_trials)
                needs_next_trial_ids = (prepare_mask & needs_next_trial_mask).nonzero(as_tuple=False).squeeze(-1)
                finished_all_trials_mask = (current_trial_counts >= max_trials)
                finished_all_trials_ids = (prepare_mask & finished_all_trials_mask).nonzero(as_tuple=False).squeeze(-1)

                if len(needs_next_trial_ids) > 0:    
                    lift_step_counts[needs_next_trial_ids] = 0
                    hold_step_counts[needs_next_trial_ids] = 0
                    pc_obs_adj, state_obs_adj = self.get_force_pc()
                    for env_id_tensor in needs_next_trial_ids:
                        env_idx = env_id_tensor.item()
                        pc_input_norm = pc_obs_adj[env_idx].unsqueeze(0)
                        state_input_norm = state_obs_adj[env_idx].unsqueeze(0)
                        obj_quat_xyzw = self.task.root_state_tensor[self.task.object_indices, 3:7]
                        object_rot_wxyz = torch.cat([obj_quat_xyzw[:, 3].unsqueeze(-1), obj_quat_xyzw[:, 0:3]], dim=1)                    
                        object_rot_matrix = pytorch3d.transforms.quaternion_to_matrix(object_rot_wxyz)
                        object_pos = self.task.root_state_tensor[self.task.object_indices, 0:3]
                        object_pose = torch.eye(4, device=self.device).unsqueeze(0).repeat(num_envs,1,1)
                        object_pose[:,0:3,0:3] = object_rot_matrix
                        object_pose[:,0:3,3] = object_pos
                        object_pose_input_norm = object_pose[env_idx].unsqueeze(0)
                        pc_np = pc_input_norm.detach().cpu().numpy()
                        state_np = state_input_norm.detach().cpu().numpy()
                        obj_pose_np = object_pose_input_norm.detach().cpu().numpy()
                        pcs_norm_np, robot_states_norm_np = normilize_batch(pc_np, state_np, obj_pose_np)
                        obs_dict = {
                            'point_cloud': pcs_norm_np[0],
                            'agent_pos': robot_states_norm_np[0 , :29]
                        }
                        result = self.dp3_policy.get_action(obs_dict)
                        corrected_states = result
                        root_state_o_corrected = corrected_states[:, 0:7]
                        dof_state_corrected = corrected_states[:, 7:]
                        pos_o = root_state_o_corrected[:, 0:3]
                        rot_o = root_state_o_corrected[:, 3:7]
                        object_state = self.task.root_state_tensor[self.task.object_indices[env_idx]].unsqueeze(0)
                        obj_pos_w = object_state[:, 0:3]
                        obj_rot_w = object_state[:, 3:7]
                        pos_o = torch.tensor(pos_o, device=self.device)
                        rot_o = torch.tensor(rot_o, device=self.device)
                        obj_pos_w = torch.tensor(obj_pos_w, device=self.device)
                        obj_rot_w = torch.tensor(obj_rot_w, device=self.device)

                        import pytorch3d.transforms as t3d
                        obj_rot_wxyz=obj_rot_w[:,[3,0,1,2]]
                        pos_w_corrected = t3d.quaternion_apply(obj_rot_wxyz, pos_o)+ obj_pos_w
                        
                        rot_o_wxyz=rot_o[:,[3,0,1,2]]
                        rot_wxyz_corrected = t3d.quaternion_multiply(obj_rot_wxyz, rot_o_wxyz)
                        rot_w_corrected = rot_wxyz_corrected[:,[1,2,3,0]]

                        root_state_w_corrected = torch.cat([pos_w_corrected, rot_w_corrected], dim=1)
                        dof_state_corrected = torch.tensor(dof_state_corrected, device=self.device)
                        corrected_states_w = torch.cat([root_state_w_corrected, dof_state_corrected], dim=1)
                        corrected_states = corrected_states_w
                        hand_idx_adjust = self.task.hand_indices[env_idx].to(torch.int32)
                        self.task.root_state_tensor[hand_idx_adjust, 0:7] = corrected_states[0, 0:7]
                        dof_pos = corrected_states[0, 7:7 + self.task.num_shadow_hand_dofs]
                        self.task.shadow_hand_dof_pos[env_idx, :] = dof_pos
                        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.task.root_state_tensor), gymtorch.unwrap_tensor(hand_idx_adjust.unsqueeze(0)), 1)
                        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.task.dof_state), gymtorch.unwrap_tensor(hand_idx_adjust.unsqueeze(0)), 1)
                    env_states[needs_next_trial_ids] = 0
                    self.gym.simulate(self.sim)
                    self.gym.fetch_results(self.sim, True)
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.task.viewer, self.task.sim, True)
                    self.task.refresh_tensors()
                    # --------error checking and reset-------#
                    object_fly_threshold_z = 1.0 
                    self.task.refresh_tensors()
                    object_positions = self.task.root_state_tensor[self.task.object_indices, 0:3]
                    object_heights = object_positions[:, 2]
                    object_fly_mask = object_heights > object_fly_threshold_z
                    if torch.any(object_fly_mask):
                        fly_env_ids = object_fly_mask.nonzero(as_tuple=False).squeeze(-1)
                        self.reset(fly_env_ids)
                    point_cloud, _ = self.get_force_pc()
                    hand_seg_id = self.task.segmentation_id['hand']
                    object_seg_id = self.task.segmentation_id['object']
                    hand_points_mask = (point_cloud[:, :, 6] == hand_seg_id)
                    object_points_mask = (point_cloud[:, :, 6] == object_seg_id)
                    hand_points_count = hand_points_mask.sum(dim=1)
                    object_points_count = object_points_mask.sum(dim=1)
                    incomplete_hand_mask = hand_points_count < 100
                    incomplete_object_mask = object_points_count < 100
                    incomplete_env_mask = incomplete_hand_mask | incomplete_object_mask

                    if torch.any(incomplete_env_mask):
                        incomplete_env_ids = incomplete_env_mask.nonzero(as_tuple=False).squeeze(-1)
                        self.reset(incomplete_env_ids)
                        env_states[incomplete_env_ids] = 0

                if len(finished_all_trials_ids) > 0:
                    initial_object_z[finished_all_trials_ids] = self.task.root_state_tensor[self.task.object_indices[finished_all_trials_ids], 2]
                    lift_step_counts[finished_all_trials_ids] = 0 
                    hold_step_counts[finished_all_trials_ids] = 0
                    env_states[finished_all_trials_ids] = 1
                    

        print(f"\n{'='*25} Evaluation finish ! {'='*25}")
        print(f"total envs: {num_envs}")
        
        total_successful_envs = torch.sum(total_successes_per_env).item()
        overall_success_rate = (total_successful_envs / num_envs) * 100
        
        print(f"success envs: {total_successful_envs}")
        print(f"success rate: {overall_success_rate:.2f}%")


    def cleanup(self):
        if self.gym:
            if self.task.viewer:
                self.gym.destroy_viewer(self.task.viewer)
            self.gym.destroy_sim(self.sim)


if __name__ == '__main__':
    class Args:
        headless = True
        cfg_env = "cfg/final.yaml"
        classifier_model_path = "final_eval_1107.pth"
        diffusion_model_path = "../Diffusion_Policy_3D/checkpoints/final_eval_w_rgb/10000.ckpt" 
        contact_map_model_path = "../gen_map/logs/final_eval_model_1/140-network.pth" 
        contact_pose_model_path = "../gen_pose/logs/final_eval_model_2/150-network.pth"
        max_adjustments = 2
        sim_device = "cuda:0"
    
    args = Args()
    evaluator = GraspEvaluator(args)
    evaluator.run_evaluation(max_adjustments=args.max_adjustments)