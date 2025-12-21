import argparse
import json
import os
import shutil

import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm
import copy
import open3d as o3d
import pickle

def main():
    parser = argparse.ArgumentParser(
        description="Convert data to zarr format for diffusion policy"
    )
    parser.add_argument(
        "data_file",
        type=str
    )
    parser.add_argument(
        "save_path",
        type=str
    )
    
    args = parser.parse_args()
    
    copy_step = 4

    
    current_abs_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_abs_dir))
    print("Project Root Dir : ", parent_dir)
    
    load_file = args.data_file
    print(f"Loading data from: {load_file}")
    
    save_dir = f"{args.save_path}.zarr"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print("Save Dir : ", save_dir)
    
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    
    # ZARR datasets will be created dynamically during the first batch write
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    # Batch processing settings
    batch_size = 100
    point_cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0
    current_batch = 0
   
    data_npz = np.load(load_file, allow_pickle=True)
    print("Loading arrays into memory...")
    fail_point_clouds_data = data_npz["fail_point_clouds"]
    fail_robot_states_data = data_npz["fail_robot_states"]
    success_robot_states_data = data_npz["success_robot_states"]
    print("Data successfully loaded into RAM.")
    train_data_num = fail_point_clouds_data.shape[0]

    for current_ep in tqdm(range(train_data_num), desc=f"Processing {train_data_num} MetaData"):
       
        for i in range(copy_step):
            point_cloud_arrays.append(fail_point_clouds_data[current_ep])
            state_arrays.append(fail_robot_states_data[current_ep])
            action_arrays.append(success_robot_states_data[current_ep])
            
            total_count += 1
        
        episode_ends_arrays.append(copy.deepcopy(total_count))
        
        # Write to ZARR if batch is full or if this is the last episode
        if (current_ep + 1) % batch_size == 0 or (current_ep + 1) == train_data_num:
            # Convert arrays to NumPy
            point_cloud_arrays = np.array(point_cloud_arrays)  # NHWC -> NCHW
            action_arrays = np.array(action_arrays)
            state_arrays = np.array(state_arrays)
            episode_ends_arrays = np.array(episode_ends_arrays)
            
            # Create datasets dynamically during the first write
            if current_batch == 0:
                zarr_data.create_dataset(
                    "point_cloud",
                    shape=(0, *point_cloud_arrays.shape[1:]),
                    chunks=(batch_size, *point_cloud_arrays.shape[1:]),
                    dtype=point_cloud_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "state",
                    shape=(0, state_arrays.shape[1]),
                    chunks=(batch_size, state_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "action",
                    shape=(0, action_arrays.shape[1]),
                    chunks=(batch_size, action_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_meta.create_dataset(
                    "episode_ends",
                    shape=(0,),
                    chunks=(batch_size,),
                    dtype="int64",
                    compressor=compressor,
                    overwrite=True,
                )
            
            # Append data to ZARR datasets
            zarr_data["point_cloud"].append(point_cloud_arrays)
            zarr_data["state"].append(state_arrays)
            zarr_data["action"].append(action_arrays)
            zarr_meta["episode_ends"].append(episode_ends_arrays)
            
            print(
                f"Batch {current_batch + 1} written with {len(point_cloud_arrays)} samples."
            )

            print(f"point_cloud shape: {point_cloud_arrays.shape}")
            print(f"state shape: {state_arrays.shape}")
            print(f"action shape: {action_arrays.shape}")
            print(f"episode_ends shape: {episode_ends_arrays.shape}")

            # Clear arrays for next batch
            point_cloud_arrays = []
            action_arrays = []
            state_arrays = []
            episode_ends_arrays = []
            current_batch += 1
            

if __name__ == "__main__":
    main()