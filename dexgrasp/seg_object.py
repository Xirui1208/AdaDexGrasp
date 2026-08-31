import os
import numpy as np
import torch
import pytorch3d.transforms as t3d
from tqdm import tqdm
from near import normilize_batch

INPUT_FILE = "DATA/AIRPLANE/AIRPLANE-success.npz"
OUTPUT_DIR = "DATA/ablation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "AIRPLANE-for2VAE.npz")
TARGET_POINTS = 1024
SEG_ID_OBJECT = 3.0
SEG_ID_CONTACT_MAP = {
    11.0: 1,  
    12.0: 2,  
    13.0: 3,  
    14.0: 4,  
    15.0: 5,  
    16.0: 6   
}
CONTACT_IDS = list(SEG_ID_CONTACT_MAP.keys())


def create_mask_0_6(object_seg_ids):
    mask_0_6 = np.zeros_like(object_seg_ids, dtype=np.int32)
    for isaac_id, cvae_label in SEG_ID_CONTACT_MAP.items():
        mask_0_6[object_seg_ids == isaac_id] = cvae_label   
    return mask_0_6

def sample_or_pad_points(points, labels, num_points):
    num_obj_points = points.shape[0]
    if num_obj_points > num_points:
        sampled_indices = np.random.choice(num_obj_points, num_points, replace=False)
        final_pc = points[sampled_indices]
        final_labels = labels[sampled_indices]
    elif num_obj_points < num_points:
        sampled_indices = np.random.choice(num_obj_points, num_points, replace=True)
        final_pc = points[sampled_indices]
        final_labels = labels[sampled_indices]
    else:
        final_pc = points
        final_labels = labels
    return final_pc, final_labels

def main():     
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    data = np.load(INPUT_FILE)
    point_clouds_world = data["point_clouds"]       # (N, P_full, 13): xyz, rgb, seg/contact ID, 6 intensities
    robot_states_world = data["robot_states"].astype(np.float32)  # (N, D_robot)
    object_states_world = data["object_states"].astype(np.float32)  # (N, 4, 4)
    
    N = point_clouds_world.shape[0]
    pcs_normalized, robot_states_normalized = normilize_batch(
        point_clouds_world, 
        robot_states_world, 
        object_states_world
    )

    pc_input_list = []
    pc_labels_list = []
    robot_state_list = [] 
    pc_with_labels_list = []
    for i in tqdm(range(N)):
        pc_norm = pcs_normalized[i] # (P_full, 13)
        seg_ids = pc_norm[:, 6]
        object_mask = (seg_ids == SEG_ID_OBJECT) #
        contact_mask = np.isin(seg_ids, CONTACT_IDS) #
        full_object_mask = object_mask | contact_mask
        object_points_xyz = pc_norm[full_object_mask, :3] # (P_obj, 3)
        object_seg_ids = seg_ids[full_object_mask]        # (P_obj,)
        
        if object_points_xyz.shape[0] == 0:
            continue
        mask_0_6 = create_mask_0_6(object_seg_ids) # (P_obj,)
        final_pc, final_labels = sample_or_pad_points(
            object_points_xyz, 
            mask_0_6, 
            TARGET_POINTS
        )
        final_labels_reshaped = np.expand_dims(final_labels, axis=-1) # (1024, 1)
        pc_with_labels = np.concatenate(
            [final_pc, final_labels_reshaped.astype(np.float32)], 
            axis=-1
        ) # (1024, 4)
        pc_input_list.append(final_pc)
        pc_labels_list.append(final_labels)
        robot_state_list.append(robot_states_normalized[i]) 
        pc_with_labels_list.append(pc_with_labels) 

    pc_input_final = np.stack(pc_input_list, axis=0)
    pc_labels_final = np.stack(pc_labels_list, axis=0)
    robot_states_final = np.stack(robot_state_list, axis=0)
    pc_with_labels_final = np.stack(pc_with_labels_list, axis=0)
    
    num_processed = pc_input_final.shape[0]    
    np.savez(
        OUTPUT_FILE,
        pc_input=pc_input_final.astype(np.float32),  
        pc_labels=pc_labels_final.astype(np.int32), 
        robot_states_normalized=robot_states_final.astype(np.float32), 
        pc_with_labels=pc_with_labels_final.astype(np.float32) 
    )
    print(f"Processed {num_processed} samples and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
   main()
