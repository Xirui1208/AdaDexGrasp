import os
import torch
import numpy as np
import open3d as o3d
import pytorch_kinematics as pk
import trimesh
import pytorch3d.transforms
import xml.etree.ElementTree as ET # 引入XML解析库
from typing import Dict

# --- 依赖说明 ---
# 确保 'csdf.py' 文件在同一目录下
from csdf import index_vertices_by_faces, compute_sdf

def parse_collision_geometry_from_urdf(urdf_path):
    """
    手动解析URDF文件, 提取所有link的<collision>信息。
    """
    print("手动解析URDF中的<collision>几何体...")
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    collision_data = {}
    
    for link in root.findall('link'):
        link_name = link.get('name')
        collisions = []
        for coll in link.findall('collision'):
            coll_info = {}
            # 解析origin
            origin = coll.find('origin')
            if origin is not None:
                xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
                rpy = [float(r) for r in origin.get('rpy', '0 0 0').split()]
            else:
                xyz, rpy = [0,0,0], [0,0,0]
            coll_info['origin_xyz'] = xyz
            coll_info['origin_rpy'] = rpy

            # 解析geometry
            geom = coll.find('geometry')
            if geom is not None:
                mesh = geom.find('mesh')
                box = geom.find('box')
                if mesh is not None:
                    coll_info['type'] = 'mesh'
                    coll_info['filename'] = mesh.get('filename')
                    scale_str = mesh.get('scale')
                    coll_info['scale'] = [float(s) for s in scale_str.split()] if scale_str else [1.0, 1.0, 1.0]
                    print("解析到碰撞网格:", coll_info['filename'], "coordinate:", coll_info['origin_xyz'], coll_info['origin_rpy'], "缩放:", coll_info['scale'])
                elif box is not None:
                    coll_info['type'] = 'box'
                    coll_info['size'] = [float(s) for s in box.get('size').split()]
            
            if 'type' in coll_info:
                collisions.append(coll_info)
        
        if collisions:
            collision_data[link_name] = collisions
            
    return collision_data


class URDFHandModelBase:

    def __init__(self, urdf_path: str, mesh_path: str, device: str = 'cpu', **kwargs):
        self.device = device
        self.mesh_path = mesh_path
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float, device=device)

        def rename_links_recurse(body):
            if hasattr(body, 'link') and body.link.name != 'world':
                body.link.name = 'robot0:' + body.link.name
            for child in body.children: rename_links_recurse(child)
        rename_links_recurse(self.chain._root)
        self.n_dofs = len(self.chain.get_joint_parameter_names())
        self.mesh = {}
        self._build_mesh_data(**kwargs)

    def _build_mesh_data(self, **kwargs):
        raise NotImplementedError

    def get_transformed_meshes(self, hand_pose: torch.Tensor) -> list:
        assert hand_pose.shape[0] == 1, "Visualization only supports a batch size of 1."
        global_translation = hand_pose[:, 0:3]
        global_rotation = pytorch3d.transforms.axis_angle_to_matrix(hand_pose[:, 3:6])
        joint_angles = hand_pose[:, 6:]
        current_status = self.chain.forward_kinematics(joint_angles)
        transformed_meshes = []
        for link_name, mesh_data in self.mesh.items():
            if not mesh_data['vertices'].numel(): continue
            matrix = current_status[link_name].get_matrix()
            local_vertices = mesh_data['vertices']
            transformed_vertices = (local_vertices @ matrix[0, :3, :3].T + matrix[0, :3, 3])
            final_vertices = (transformed_vertices @ global_rotation[0].T + global_translation[0])
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(final_vertices.cpu().numpy())
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_data['faces'].cpu().numpy())
            o3d_mesh.compute_vertex_normals()
            transformed_meshes.append(o3d_mesh)
        return transformed_meshes

class URDFHandVisualModel(URDFHandModelBase):
    def _build_mesh_data(self, **kwargs):
        def build_mesh_recurse(body):
            if hasattr(body, 'link') and len(body.link.visuals) > 0:
                link_name = body.link.name
                if link_name == 'robot0:world':
                    for child in body.children: build_mesh_recurse(child)
                    return
                link_vertices, link_faces, n_link_vertices = [], [], 0
                for visual in body.link.visuals:
                    if visual.geom_type == "mesh":
                        scale_vec = None
                        if "tactile" in visual.geom_param:
                            filename = visual.geom_param[3:]
                            if "palm" in filename:
                                scale_vec = [1.0, 1.0, 1.0]
                            else:
                                scale_vec = [0.3, 0.3, 0.3]
                        else:
                            filename = visual.geom_param
                        if scale_vec is None: 
                            scale_vec = [1.0, 1.0, 1.0]
                        full_mesh_path = os.path.join(self.mesh_path, filename.replace('../', ''))
                        try: link_mesh = trimesh.load_mesh(full_mesh_path, process=False)
                        except Exception:
                            print("无法加载visual网格:", full_mesh_path) 
                            continue
                        vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=self.device)
                        faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=self.device)
                        pos = visual.offset.to(self.device)
                        vertices = vertices * torch.tensor(scale_vec, dtype=torch.float, device=self.device)
                        vertices = pos.transform_points(vertices)
                        link_vertices.append(vertices)
                        link_faces.append(faces + n_link_vertices)
                        n_link_vertices += len(vertices)
                if link_vertices: self.mesh[link_name] = {'vertices': torch.cat(link_vertices, dim=0), 'faces': torch.cat(link_faces, dim=0)}
            for child in body.children: build_mesh_recurse(child)
        build_mesh_recurse(self.chain._root)

class URDFHandCollisionModel(URDFHandModelBase):
    def _build_mesh_data(self, collision_data=None, **kwargs):

        def build_mesh_recurse(body):
            if hasattr(body, 'link'):
                link_name_with_prefix = body.link.name
                link_name = link_name_with_prefix.replace('robot0:', '')
                
                if link_name in collision_data:
                    link_vertices, link_faces, n_link_vertices = [], [], 0
                    for coll_info in collision_data[link_name]:
                        link_mesh = None
                        if coll_info['type'] == 'mesh':
                            full_mesh_path = os.path.join(self.mesh_path, coll_info['filename'].replace('../', ''))
                            try: 
                                link_mesh = trimesh.load_mesh(full_mesh_path, process=False)
                                if "tactile" in coll_info['filename']:
                                    print(coll_info['filename'])
                                    scale_vector = coll_info.get('scale', [0.3, 0.3, 0.3])
                                    link_mesh.apply_scale(scale_vector)
                            except Exception:
                                print("无法加载碰撞网格:", full_mesh_path)
                                continue
                        elif coll_info['type'] == 'box':
                            link_mesh = trimesh.primitives.Box(extents=coll_info['size'])
                        
                        if link_mesh:
                            vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=self.device)
                            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=self.device)
                            origin_xyz = torch.tensor(coll_info['origin_xyz'], device=self.device)
                            origin_rpy = torch.tensor(coll_info['origin_rpy'], device=self.device)
                            origin_rot = pytorch3d.transforms.euler_angles_to_matrix(origin_rpy, "XYZ")
                            vertices = vertices @ origin_rot.T + origin_xyz
                            link_vertices.append(vertices)
                            link_faces.append(faces + n_link_vertices)
                            n_link_vertices += len(vertices)
                    
                    if link_vertices:
                        self.mesh[link_name_with_prefix] = {'vertices': torch.cat(link_vertices, dim=0), 'faces': torch.cat(link_faces, dim=0)}
                        self.mesh[link_name_with_prefix]['face_verts'] = index_vertices_by_faces(self.mesh[link_name_with_prefix]['vertices'], self.mesh[link_name_with_prefix]['faces'])

            for child in body.children:
                build_mesh_recurse(child)
        build_mesh_recurse(self.chain._root)

    def get_signed_distances(self, hand_pose: torch.Tensor, object_pc: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculates the signed distances from object points to each collision link.

        Returns:
            A dictionary mapping link_name (str) to its distance tensor (torch.Tensor of shape [B, N]).
        """
        batch_size = hand_pose.shape[0]
        global_translation, global_rotation = hand_pose[:, 0:3], pytorch3d.transforms.axis_angle_to_matrix(hand_pose[:, 3:6])
        joint_angles = hand_pose[:, 6:]
        # print("Joint angles shape:", joint_angles.shape)

        # print("Joint Names:",self.chain.get_joint_parameter_names())
        joint_names=['FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1']
        joint_angles_dict={name:joint_angles[:,i] for i,name in enumerate(joint_names)}
        current_status = self.chain.forward_kinematics(joint_angles_dict)
        # print("Current_status:", current_status)
        # Transform object points into the hand's root frame
        x = (object_pc - global_translation.unsqueeze(1)) @ global_rotation
        
        per_link_distances = {}
        for link_name in self.mesh:
            # print("mesh:",self.mesh[link_name] )
            if 'face_verts' not in self.mesh[link_name] or not self.mesh[link_name]['face_verts'].numel():
                continue
            
            # Transform object points into the link's local frame
            matrix = current_status[link_name].get_matrix()

            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            
            # Compute signed distance function
            dis_local_sq, _, dis_signs, _, _ = compute_sdf(x_local.reshape(-1, 3), self.mesh[link_name]['face_verts'])
            
            # Get signed distance (negative outside, positive inside)
            # dis_local = dis_local_sq.sqrt() * (-dis_signs)
            
            per_link_distances[link_name] = dis_local_sq.reshape(batch_size, -1)
            
        return per_link_distances,x

def visualize_contact_points():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    URDF_FILE_PATH = '../../assets/urdf/shadow_hand_tacsl/robots/movable_shadowhand_collision.urdf'
    MESH_BASE_PATH = '../../assets/urdf/shadow_hand_tacsl'

    MIN_CONTACTS_PER_SENSOR = 10
    TACTILE_DISTANCE_THRESHOLD = 4e-6
    GLOBAL_DISTANCE_THRESHOLD = 2e-6
    TACTILE_LINK_NAMES = [
        'robot0:palm_gelsight',
        'robot0:thumb_gelsight',
        'robot0:ff_gelsight',
        'robot0:mf_gelsight',
        'robot0:rf_gelsight',
        'robot0:lf_gelsight',
    ]
    collision_data = parse_collision_geometry_from_urdf(URDF_FILE_PATH)
    collision_model = URDFHandCollisionModel(urdf_path=URDF_FILE_PATH, mesh_path=MESH_BASE_PATH, device=DEVICE, collision_data=collision_data)
    visual_model = URDFHandVisualModel(urdf_path=URDF_FILE_PATH, mesh_path=MESH_BASE_PATH, device=DEVICE)
    hand_pose = torch.zeros(1, 6 + visual_model.n_dofs, device=DEVICE)
    hand_pose[0, 6 + 1] = 1.2; hand_pose[0, 6 + 2] = 1.2; hand_pose[0, 6 + 3] = 0.5
    hand_pose[0, 6 + 5] = 1.2; hand_pose[0, 6 + 6] = 1.2; hand_pose[0, 6 + 7] = 0.5
    hand_pose[0, 6 + 9] = 1.2; hand_pose[0, 6 + 10] = 1.2; hand_pose[0, 6 + 11] = 0.5
    hand_pose[0, 6 + 17] = 0.6; hand_pose[0, 6 + 18] = 0.8; hand_pose[0, 6 + 20] = 0.7

    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    sphere_mesh.translate((-0.032, -0.027, 0.17))
    object_pcd_o3d = sphere_mesh.sample_points_uniformly(number_of_points=2000)
    object_pc_np = np.asarray(object_pcd_o3d.points)
    object_pc_torch = torch.from_numpy(object_pc_np).float().unsqueeze(0).to(DEVICE)

    per_link_distances = collision_model.get_signed_distances(hand_pose, object_pc_torch)
    all_distances_stacked = torch.stack([d.abs() for d in per_link_distances.values()])
    global_min_distances = torch.min(all_distances_stacked, dim=0)[0].squeeze(0)
    global_contact_mask = global_min_distances < GLOBAL_DISTANCE_THRESHOLD
    num_points = object_pc_torch.shape[1]
    tactile_contact_mask = torch.zeros(num_points, dtype=torch.bool, device=DEVICE)
    
    for link_name in TACTILE_LINK_NAMES:
        if link_name not in per_link_distances:
            continue
        
        link_dist = per_link_distances[link_name].squeeze(0).abs()
        close_points_mask = link_dist < TACTILE_DISTANCE_THRESHOLD
        num_close_points = torch.sum(close_points_mask)
        current_contact_indices = torch.where(close_points_mask)[0]

        if current_contact_indices.numel() > 0:
            tactile_contact_mask[current_contact_indices] = True

    total_contact_mask = tactile_contact_mask | global_contact_mask
    colors = np.full_like(object_pc_np, [0.5, 0.5, 0.5])
    colors[total_contact_mask.cpu().numpy()] = [1.0, 0, 0]
    object_pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    hand_meshes = visual_model.get_transformed_meshes(hand_pose)
    for mesh in hand_meshes: mesh.paint_uniform_color([0.7, 0.7, 0.8])
    collision_meshes = collision_model.get_transformed_meshes(hand_pose)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Combined Contact Visualization", width=1024, height=768)
    vis.add_geometry(object_pcd_o3d)
    for mesh in hand_meshes: vis.add_geometry(mesh)
    for mesh in collision_meshes: vis.add_geometry(mesh)
    render_option = vis.get_render_option()
    render_option.mesh_show_wireframe = True
    render_option.background_color = np.asarray([0.1, 0.1, 0.1])
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    visualize_contact_points()
