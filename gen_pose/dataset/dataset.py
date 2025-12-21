import torch
import numpy as np
import os
from torch.utils.data import Dataset

    

class RegressionGraspDataset(Dataset):
    def __init__(self,dir,split,train_num:int=7680,val_num=640, data_file_name:str="data.npz"):
        self.dir=dir
        self.data_file_path = os.path.join(self.dir, data_file_name)
        self.data = self.make_dataset(split,train_num,val_num)

    
    def __len__(self):
        return len(self.data["inputs"])

    def __getitem__(self, idx):
        point_cloud_with_mask = self.data["inputs"][idx]
        robot_state = self.data["targets"][idx]       
        
        if not isinstance(point_cloud_with_mask, torch.Tensor):
            point_cloud_with_mask = torch.from_numpy(point_cloud_with_mask).float()
        
        if not isinstance(robot_state, torch.Tensor):
            robot_state = torch.from_numpy(robot_state).float()
            
        return point_cloud_with_mask, robot_state

    def make_dataset(self,split,train_num,val_num)->dict:
        dataset = {}
        print(f"Loading preprocessed regression data from '{self.data_file_path}'...")
        _data = np.load(self.data_file_path, allow_pickle=True)
        inputs = _data["pc_with_labels"].astype(np.float32)
        targets = _data["robot_states_normalized"].astype(np.float32)
        total_samples = len(inputs)
        print(f"Loaded {total_samples} total samples.")
        if split == "train":
            dataset = {
                "inputs": inputs[:train_num],
                "targets": targets[:train_num]
            }
            print(f"Using {len(dataset['inputs'])} samples for training.")
        
        elif split == "val":
            dataset = {
                "inputs": inputs[train_num : train_num + val_num],
                "targets": targets[train_num : train_num + val_num]
            }
            print(f"Using {len(dataset['inputs'])} samples for validation.")
        
        else:
            raise ValueError(f"unknown split: {split}")

        return dataset