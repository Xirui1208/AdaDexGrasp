import os
import numpy as np
import torch
import pytorch3d.transforms as t3d

DATA_PATH = "total.npz" 
OUT_DIR = "DATA"
OUT_NPZ = os.path.join(OUT_DIR, "adapt-data.npz")
ESP = torch.finfo().eps * 4

def poses_to_matrices(pos, quat_xyzw):
    B = pos.shape[0]
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    R = t3d.quaternion_to_matrix(quat_wxyz)  # (B, 3, 3)
    T = torch.eye(4, device=pos.device, dtype=pos.dtype).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    return T

def geodesic_distance_cdist(R1, R2):
    R1_T = R1.transpose(1, 2).unsqueeze(1)
    R2_b = R2.unsqueeze(0)
    R = torch.matmul(R1_T, R2_b)
    diagonals = torch.diagonal(R, dim1=-2, dim2=-1)
    traces = torch.sum(diagonals, dim=-1)
    theta = torch.clamp(0.5 * (traces - 1), -1 + ESP, 1 - ESP)
    dist = torch.acos(theta)
    return dist

def double_geodesic_distance_cdist(T1, T2):
    R1, t1 = T1[:, :3, :3], T1[:, :3, 3]  # (N, 3, 3), (N, 3)
    R2, t2 = T2[:, :3, :3], T2[:, :3, 3]  # (M, 3, 3), (M, 3)
    dist_R = geodesic_distance_cdist(R1, R2)
    dist_t = torch.cdist(t1, t2, p=2.0)
    return dist_t, dist_R

def compute_physical_distance(states_fail, states_succ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    succ_t = torch.from_numpy(states_succ).to(device=device, dtype=torch.float32)
    pos_succ = succ_t[:, 0:3]
    quat_succ = succ_t[:, 3:7]
    dof_pos_succ = succ_t[:, 7:7+22]
    T_succ = poses_to_matrices(pos_succ, quat_succ) # (Ns, 4, 4)
    Nf = states_fail.shape[0]
    Ns = states_succ.shape[0]
    BATCH_SIZE = 1024 
    
    all_total_dists = []
    for i in range(0, Nf, BATCH_SIZE):
        end = min(i + BATCH_SIZE, Nf)
        fail_batch_np = states_fail[i:end]
        fail_t = torch.from_numpy(fail_batch_np).to(device=device, dtype=torch.float32)
        pos_fail = fail_t[:, 0:3]
        quat_fail = fail_t[:, 3:7]
        dof_pos_fail = fail_t[:, 7:7+22]
        T_fail_batch = poses_to_matrices(pos_fail, quat_fail)
        dist_trans, dist_rot = double_geodesic_distance_cdist(T_fail_batch, T_succ)
        dist_pose = torch.sqrt(dist_trans**2 + dist_rot**2)
        dist_dof = torch.cdist(dof_pos_fail, dof_pos_succ, p=2.0)
        total_dist_batch = dist_pose*10 + dist_dof
        all_total_dists.append(total_dist_batch.detach().cpu())
    final_dist_matrix = torch.cat(all_total_dists, dim=0)
    return final_dist_matrix.numpy()

def normilize_batch(pcs:np.array, robot_states:np.array,object_states:np.array):
    N = pcs.shape[0]
    pcs_normalized = pcs.copy()
    robot_pose_normalized = robot_states[:, :7].copy()
    object_positions = object_states[:, :3, 3]  # (N, 3)
    object_orientations_matrix = object_states[:, :3, :3]  # (N, 3, 3)
    pcs_normalized[:, :, :3] -= object_positions[:, np.newaxis, :]
    robot_pose_normalized[:, :3] -= object_positions
    R_O_W = np.transpose(object_orientations_matrix, (0, 2, 1))
    pcs_normalized[:, :, :3] = np.einsum('nij,npj->npi', R_O_W, pcs_normalized[:, :, :3])
    robot_pose_normalized[:, :3] = np.einsum('nij,nj->ni', R_O_W, robot_pose_normalized[:, :3])
    robot_quat = robot_pose_normalized[:, 3:7]  # (N, 4)
    robot_quat_wxyz=np.concatenate([robot_quat[:, 3:4], robot_quat[:, 0:3]], axis=1)
    robot_quat_tensor_wxyz = torch.from_numpy(robot_quat_wxyz).to(dtype=torch.float32)
    R_W_R = t3d.quaternion_to_matrix(robot_quat_tensor_wxyz).numpy()  # (N, 3, 3)
    R_O_R = np.matmul(R_O_W, R_W_R)
    R_O_R_tensor = torch.from_numpy(R_O_R).to(dtype=torch.float32)
    q_O_R_tensor_wxyz = t3d.matrix_to_quaternion(R_O_R_tensor)
    q_O_R_wxyz = q_O_R_tensor_wxyz.numpy()
    q_O_R_xyzw = np.concatenate([q_O_R_wxyz[:, 1:4], q_O_R_wxyz[:, 0:1]], axis=1)
    robot_pose_normalized[:, 3:7] = q_O_R_xyzw
    robot_states_normalized = robot_states.copy()
    robot_states_normalized[:, :7] = robot_pose_normalized
    return pcs_normalized, robot_states_normalized

def main():
    data = np.load(DATA_PATH)
    pcs = data["point_clouds"]
    robot_states = data["robot_states"].astype(np.float32)
    labels = data["labels"].astype(np.int32)
    object_states = data["object_states"].astype(np.float32)
    N = pcs.shape[0]
    num_succ = int((labels == 1).sum())
    num_fail = int((labels == 0).sum())
    pcs_normalized, robot_states_normalized = normilize_batch(pcs, robot_states, object_states)
    idx_succ = np.where(labels == 1)[0]
    idx_fail = np.where(labels == 0)[0]
    states_succ = robot_states_normalized[idx_succ]
    states_fail = robot_states_normalized[idx_fail]
    dist_matrix = compute_physical_distance(states_fail, states_succ)
    k = 1
    nn_indices_in_succ_array = np.argsort(dist_matrix, axis=1)[:, :k]  # (Nf, k)
    nn_global_indices = idx_succ[nn_indices_in_succ_array]  # (Nf, k)
    conditioning_pcs = pcs_normalized[idx_fail]
    conditioning_states = states_fail
    final_fail_pcs = np.repeat(conditioning_pcs, k, axis=0)
    final_fail_states = np.repeat(conditioning_states, k, axis=0)
    success_pcs = pcs_normalized[nn_global_indices]
    success_states = robot_states_normalized[nn_global_indices]
    final_success_pcs = success_pcs.reshape(-1, pcs_normalized.shape[1], pcs_normalized.shape[2])
    final_success_states = success_states.reshape(-1, robot_states_normalized.shape[1])
    expected_size = num_fail * k
    assert final_fail_pcs.shape[0] == expected_size
    assert final_fail_states.shape[0] == expected_size
    assert final_success_pcs.shape[0] == expected_size
    assert final_success_states.shape[0] == expected_size
    np.savez(
        OUT_NPZ,
        fail_point_clouds=final_fail_pcs,
        fail_robot_states=final_fail_states,
        success_point_clouds=final_success_pcs,
        success_robot_states=final_success_states
    )
    print(f"  - fail_point_clouds: {final_fail_pcs.shape}")
    print(f"  - fail_robot_states: {final_fail_states.shape}")
    print(f"  - success_point_clouds: {final_success_pcs.shape}")
    print(f"  - success_robot_states: {final_success_states.shape}")

if __name__ == "__main__":
   main()