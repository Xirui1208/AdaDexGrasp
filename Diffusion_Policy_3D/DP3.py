import os
import pathlib
import pdb

import dill
import hydra
import torch
from omegaconf import OmegaConf

from Diffusion_Policy_3D.train import TrainDP3Workspace


class DP3:
    def __init__(self, checkpoint_path, device="cuda:0") -> None:
        # load checkpoint
        checkpoint = checkpoint_path
        task_name = checkpoint.split("/")[-2]
        checkpoint_num = int(checkpoint.split("/")[-1].split(".")[0])
        print(f"Loading DP3 policy from checkpoint: {checkpoint} (task: {task_name}, step: {checkpoint_num})")
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        self.policy, self.env_runner = self.get_policy_and_runner(cfg, checkpoint_path)
        self.policy.to(device)
        self.policy.eval()

    def update_obs(self, observation):
        self.env_runner.update_obs(observation)

    def get_action(self, observation):
        action = self.env_runner.get_action(self.policy, observation)
        return action

    def get_policy_and_runner(self, cfg, checkpoint_path):
        workspace = TrainDP3Workspace(cfg)
        policy, env_runner = workspace.get_policy_and_runner(cfg, checkpoint_path)
        return policy, env_runner



