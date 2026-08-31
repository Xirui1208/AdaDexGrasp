from tasks.tacsl_sensors.tacsl_sensors import CameraSensor, TactileRGBSensor, TactileFieldSensor
import torch

class TacSLSensors(TactileFieldSensor, TactileRGBSensor, CameraSensor):

    def get_regular_camera_specs(self):
        # TODO: Maybe this should go inside the task files, as it depends on task configs e.g. self.cfg_task.env.use_camera

        camera_spec_dict = {}
        # if self.cfg["tacsl"]["use_camera"]:
            # camera_spec_dict = {c_cfg["name"]: c_cfg for c_cfg in self.cfg["tacsl"]["camera_configs"]}
        return camera_spec_dict


    def _compose_tactile_image_configs(self,actor_handle):
        configs = []
        for name, attach_link, elastomer_link in [
            ('palm_tactile_camera', 'palm_tip',  'palm_gelsight'),
            ('thumb_tactile_camera', 'thumb_tip', 'thumb_gelsight'),
            ('ff_tactile_camera', 'ff_tip',    'ff_gelsight'),
            ('mf_tactile_camera', 'mf_tip',    'mf_gelsight'),
            ('rf_tactile_camera', 'rf_tip',    'rf_gelsight'),
            ('lf_tactile_camera', 'lf_tip',    'lf_gelsight'),
        ]:
            cfg = dict(
                tactile_camera_name=name,
                actor_name='hand',
                actor_handle=actor_handle,
                attach_link_name=attach_link,
                elastomer_link_name=elastomer_link,
                compliance_stiffness=self.cfg["tacsl"]["compliance_stiffness"],
                compliant_damping=self.cfg["tacsl"]["compliant_damping"],
                use_acceleration_spring=False,
                sensor_type='gelsight_r15'
            )
            configs.append(cfg)
        return configs

    def _compose_tactile_force_field_configs(self,actor_handle,object_name,asset_path):
        configs = []
        for name, elastomer, elastomer_tip in [
            ('palm_force_field',  'palm_gelsight', 'palm_tip'),
            ('thumb_force_field', 'thumb_gelsight', 'thumb_tip'),
            ('ff_force_field',    'ff_gelsight', 'ff_tip'),
            ('mf_force_field',    'mf_gelsight', 'mf_tip'),
            ('rf_force_field',    'rf_gelsight', 'rf_tip'),
            ('lf_force_field',    'lf_gelsight', 'lf_tip'),
        ]:
            cfg = dict([
                ('name', name),
                ('elastomer_actor_name', 'hand'), ('elastomer_link_name', elastomer),
                ('elastomer_tip_link_name', elastomer_tip),
                ('elastomer_parent_urdf_path', asset_path["shadow_hand"]),
                ('indenter_urdf_path', asset_path["object"]),
                ('indenter_actor_name', 'object'), ('indenter_link_name', object_name[0]),
                ('actor_handle', actor_handle),
                ('compliance_stiffness', self.cfg["tacsl"]["compliance_stiffness"]),
                ('compliant_damping', self.cfg["tacsl"]["compliant_damping"]),
                ('use_acceleration_spring', False)
            ])
            configs.append(cfg)
        return configs
    
    def get_tactile_force_field_tensors_dict(self):

        tactile_force_field_dict_raw = self.get_tactile_shear_force_fields()
        tactile_force_field_dict_processed = dict()
        nrows, ncols = self.cfg["tacsl"]["num_shear_rows"], self.cfg["tacsl"]["num_shear_cols"]

        debug = False   # Debug visualization
        for k in tactile_force_field_dict_raw:
            penetration_depth, tactile_normal_force, tactile_shear_force = tactile_force_field_dict_raw[k]
            tactile_force_field = torch.cat(
                (tactile_normal_force.view((self.num_envs, nrows, ncols, 1)),
                 tactile_shear_force.view((self.num_envs, nrows, ncols, 2))),
                dim=-1)
            tactile_force_field_dict_processed[k] = tactile_force_field

            if debug:
                env_viz_id = 0
                tactile_image = visualize_tactile_shear_image(
                    tactile_normal_force[env_viz_id].view((nrows, ncols)).cpu().numpy(),
                    tactile_shear_force[env_viz_id].view((nrows, ncols, 2)).cpu().numpy(),
                    normal_force_threshold=0.0008,
                    shear_force_threshold=0.0008)
                cv2.imshow(f'Force Field {k}', tactile_image.swapaxes(0, 1))

                penetration_depth_viz = visualize_penetration_depth(
                    penetration_depth[env_viz_id].view((nrows, ncols)).cpu().numpy(),
                    resolution=5, depth_multiplier=300.)
                cv2.imshow(f'FF Penetration Depth {k}', penetration_depth_viz.swapaxes(0, 1))
        return tactile_force_field_dict_processed

    def _create_sensors(self,actor_handle,object_rb_names,asset_path):
        self.camera_spec_dict = dict()
        self.camera_handles_list = []
        self.camera_tensors_list = []
        if self.cfg["tacsl"]["use_isaac_gym_tactile"]:
            tactile_sensor_configs = self._compose_tactile_image_configs(actor_handle)
            self.set_compliant_dynamics_for_tactile_sensors(tactile_sensor_configs)
            camera_spec_dict_tactile = self.get_tactile_rgb_camera_configs(tactile_sensor_configs)
            self.camera_spec_dict.update(camera_spec_dict_tactile)
        if self.cfg["tacsl"]["use_camera"]:
            camera_spec_dict = self.get_regular_camera_specs()
            self.camera_spec_dict.update(camera_spec_dict)
        if self.camera_spec_dict:
            # tactile cameras created along with other cameras in create_camera_actors
            camera_handles_list, camera_tensors_list = self.create_camera_actors(self.camera_spec_dict)
            self.camera_handles_list += camera_handles_list
            self.camera_tensors_list += camera_tensors_list

        if self.cfg["tacsl"]["use_shear_force"]:
            tactile_ff_configs = self._compose_tactile_force_field_configs(actor_handle,object_rb_names,asset_path)
            self.set_compliant_dynamics_for_tactile_sensors(tactile_ff_configs)
            self.sdf_tool = 'physx'
            self.sdf_tensor = self.setup_tactile_force_field(self.sdf_tool,
                                                             self.cfg["tacsl"]["num_shear_rows"],
                                                             self.cfg["tacsl"]["num_shear_cols"],
                                                             tactile_ff_configs)
