import torch
import numpy as np
import os
from torch.utils.data import Dataset

    

class GraspDataset(Dataset):
    def __init__(self,dir,split,train_num:int=7680,val_num=640, data_file_name:str="data.npz"):
        self.dir=dir
        self.data_file_path = os.path.join(self.dir, data_file_name)
        self.data = self.make_dataset(split,train_num,val_num)

    def __len__(self):
        return len(self.data["pc_input"])

    def __getitem__(self, idx):
        point_clouds = self.data["pc_input"][idx][:, :3]
        labels = self.data["pc_labels"][idx]       # (1024,)
        return point_clouds, labels


    def make_dataset(self,split,train_num,val_num)->dict:
        dataset = {}
        print(f"Loading preprocessed segmentation data from '{self.data_file_path}'...")
        _data = np.load(self.data_file_path, allow_pickle=True)
        point_clouds = _data["pc_input"]
        labels = _data["pc_labels"]
        point_clouds = torch.from_numpy(point_clouds).float()
        labels = torch.from_numpy(labels).long()
        total_samples = len(point_clouds)
        print(f"Loaded {total_samples} total samples.")
        if split == "train":
            dataset = {
                "pc_input": point_clouds[:train_num],
                "pc_labels": labels[:train_num]
            }
            print(f"Using {len(dataset['pc_input'])} samples for training.")
        
        elif split == "val":
            dataset = {
                "pc_input": point_clouds[train_num : train_num + val_num],
                "pc_labels": labels[train_num : train_num + val_num]
            }
            print(f"Using {len(dataset['pc_input'])} samples for validation.")
        
        else:
            raise ValueError(f"unknown split: {split}")

        return dataset