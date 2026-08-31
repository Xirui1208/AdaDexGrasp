import torch
import os
import numpy as np


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def get_unique_filename(base_filename, extension):
    counter = 0
    filename = f"{base_filename}_{counter}{extension}"
    while os.path.exists(filename):
        counter += 1
        filename = f"{base_filename}_{counter}{extension}"

    return filename


def get_unique_dirname(dirname):
    counter = 0
    new_dirname = f"{dirname}_{counter}"
    while os.path.exists(new_dirname):
        counter += 1
        new_dirname = f"{dirname}_{counter}"

    return new_dirname


def has_bad_grad(model):
    for p in model.parameters():
        if p.grad is not None and (
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
        ):
            return True
    return False
