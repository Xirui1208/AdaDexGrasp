import os
import numpy as np
import torch
import pytorch3d.transforms as t3d
from tqdm import tqdm

# --- 检查 near.py 是否存在 ---
try:
    from near import normilize_batch
except ImportError:
    print("="*50)
    print("!! 错误: 无法导入 'from near import normilize_batch'。")
    print(f"!! 请确保 'near.py' 文件 与此脚本 'seg_object.py' 位于同一目录中。")
    print("="*50)
    exit()

# --- 配置 ---
# 1. 输入文件 (来自 shadow_hand_random_load_vision_adapt.py)
INPUT_FILE = "DATA/AIRPLANE/AIRPLANE-success.npz"

# 2. 输出文件 (用于 CVAE 的 train.py 和 dataset.py)
OUTPUT_DIR = "DATA/ablation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "AIRPLANE-for2VAE.npz")

# 3. PointNet++ 的固定点数
TARGET_POINTS = 1024

# 4. 分割ID映射
# 来自 shadow_hand_random_load_vision_adapt.py
SEG_ID_OBJECT = 3.0
SEG_ID_CONTACT_MAP = {
    11.0: 1,  # 大拇指 -> Class 1
    12.0: 1,  # 食指   -> Class 2
    13.0: 1,  # 中指   -> Class 3
    14.0: 1,  # 無名指 -> Class 4
    15.0: 1,  # 小指   -> Class 5
    16.0: 1   # 掌心   -> Class 6
}
CONTACT_IDS = list(SEG_ID_CONTACT_MAP.keys())


def create_mask_0_6(object_seg_ids):
    """
    将 Isaac Gym 的 seg_id (3.0, 11.0-16.0) 映射到 CVAE 的标签 (0-6)。
    """
    # 1. 默认为 0 (非接触)
    mask_0_6 = np.zeros_like(object_seg_ids, dtype=np.int32)
    
    # 2. 循环映射 11-16 到 1-6
    for isaac_id, cvae_label in SEG_ID_CONTACT_MAP.items():
        mask_0_6[object_seg_ids == isaac_id] = 1
        
    return mask_0_6

def sample_or_pad_points(points, labels, num_points):
    """
    将点云和标签采样或填充到固定数量 'num_points'。
    """
    num_obj_points = points.shape[0]
    
    if num_obj_points > num_points:
        # 随机下采样
        sampled_indices = np.random.choice(num_obj_points, num_points, replace=False)
        final_pc = points[sampled_indices]
        final_labels = labels[sampled_indices]
    elif num_obj_points < num_points:
        # 随机上采样 (带放回)
        sampled_indices = np.random.choice(num_obj_points, num_points, replace=True)
        final_pc = points[sampled_indices]
        final_labels = labels[sampled_indices]
    else:
        # 数量刚刚好
        final_pc = points
        final_labels = labels
        
    return final_pc, final_labels

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"输入数据文件未找到: {INPUT_FILE}")
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 加载原始数据
    print(f"正在加载原始数据从: {INPUT_FILE}")
    data = np.load(INPUT_FILE)
    point_clouds_world = data["point_clouds"]       # (N, P_full, 7)
    robot_states_world = data["robot_states"].astype(np.float32)  # (N, D_robot)
    object_states_world = data["object_states"].astype(np.float32)  # (N, 4, 4)
    
    N = point_clouds_world.shape[0]
    print(f"加载了 {N} 个样本。")

    # 2. 【关键】使用 near.py 的函数进行归一化
    print("正在使用 'normilize_batch' 归一化所有点云和状态到物体坐标系...")
    pcs_normalized, robot_states_normalized = normilize_batch(
        point_clouds_world, 
        robot_states_world, 
        object_states_world
    )
    # pcs_normalized shape is (N, P_full, 7)
    # robot_states_normalized shape is (N, D_robot)
    print("归一化完成。")

    pc_input_list = []
    pc_labels_list = []
    robot_state_list = [] 
    pc_with_labels_list = [] # <-- (新增) 用于存储合并后的 (N, 1024, 4)

    print(f"正在处理 {N} 个样本 (提取物体点, 创建 0-6 遮罩, 采样到 {TARGET_POINTS} 点)...")
    for i in tqdm(range(N)):
        pc_norm = pcs_normalized[i] # (P_full, 7)
        seg_ids = pc_norm[:, 6]

        # 3. 提取所有物体点 (包括接触点)
        object_mask = (seg_ids == SEG_ID_OBJECT) #
        contact_mask = np.isin(seg_ids, CONTACT_IDS) #
        
        # 确认 "接触的也是物体"
        full_object_mask = object_mask | contact_mask
        
        object_points_xyz = pc_norm[full_object_mask, :3] # (P_obj, 3)
        object_seg_ids = seg_ids[full_object_mask]        # (P_obj,)
        
        if object_points_xyz.shape[0] == 0:
            continue

        # 4. 创建 0-6 遮罩
        mask_0_6 = create_mask_0_6(object_seg_ids) # (P_obj,)
        
        # 5. 采样或填充到 TARGET_POINTS
        final_pc, final_labels = sample_or_pad_points(
            object_points_xyz, 
            mask_0_6, 
            TARGET_POINTS
        )
        # final_pc shape: (1024, 3)
        # final_labels shape: (1024,)
        
        # --- (新增) 6. 创建合并后的 (XYZ + Label) 数组 ---
        final_labels_reshaped = np.expand_dims(final_labels, axis=-1) # (1024, 1)
        pc_with_labels = np.concatenate(
            [final_pc, final_labels_reshaped.astype(np.float32)], 
            axis=-1
        ) # (1024, 4)
        
        # 7. 添加到各自的列表中
        pc_input_list.append(final_pc)
        pc_labels_list.append(final_labels)
        robot_state_list.append(robot_states_normalized[i]) 
        pc_with_labels_list.append(pc_with_labels) # <-- (新增)

    # 8. 最终保存
    pc_input_final = np.stack(pc_input_list, axis=0)
    pc_labels_final = np.stack(pc_labels_list, axis=0)
    robot_states_final = np.stack(robot_state_list, axis=0)
    pc_with_labels_final = np.stack(pc_with_labels_list, axis=0) # <-- (新增)
    
    num_processed = pc_input_final.shape[0]
    print(f"处理完成。总共 {num_processed}/{N} 个有效样本被保存。")
    
    # 保存为 CVAE dataset.py 可以读取的格式
    np.savez(
        OUTPUT_FILE,
        pc_input=pc_input_final.astype(np.float32),     # (N_proc, 1024, 3)
        pc_labels=pc_labels_final.astype(np.int32),     # (N_proc, 1024)
        robot_states_normalized=robot_states_final.astype(np.float32), # (N_proc, D_robot)
        
        # --- (新增) 保存合并后的 (XYZ + Label) ---
        pc_with_labels=pc_with_labels_final.astype(np.float32) # (N_proc, 1024, 4)
    )
    
    print(f"\n成功将数据保存到: {OUTPUT_FILE}")
    print("文件内容 (已归一化):")
    print(f"  - pc_input: {pc_input_final.shape} (用于 CVAE 输入, 仅 XYZ)")
    print(f"  - pc_labels: {pc_labels_final.shape} (用于 CVAE 标签, 仅 0-6 遮罩)")
    print(f"  - robot_states_normalized: {robot_states_final.shape} (归一化的机器人状态)")
    print(f"  - pc_with_labels: {pc_with_labels_final.shape} (新增: XYZ + 0-6 遮罩)")


if __name__ == "__main__":
   main()