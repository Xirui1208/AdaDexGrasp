import os
import time
import yaml
import sys
import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm, trange
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from model import Network
from dataset.dataset import GraspDataset as Dataset
from utils.wandb import log_init, log_writer
from utils.train_utils import optimizer_to_device, get_unique_dirname


def train(task, config, data_dir="data", base_dir="logs"):
    val_every_epoch = config["val_every_epoch"]
    save_every_epoch = config["save_every_epoch"]

    train_num = config["train_num"]
    val_num = config["val_num"]

    save_dir_base = f"{base_dir}/{task}"
    save_dir = get_unique_dirname(save_dir_base)

    point_cloud_dim = config["point_cloud_dim"] 
    num_classes = config["num_classes"]
    feat_dim = config["feat_dim"]
    
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    lr_decay_every = config["lr_decay_every"]
    lr_decay_by = config["lr_decay_by"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = random.randint(1, 10000) if "seed" not in config else config["seed"]

    config["seed"] = seed
    log_init(run_name=task , cfg=config, mode="online")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Creating network (for Segmentation Task) ...... ")
    network = Network(
        point_cloud_dim=point_cloud_dim,
        num_classes=num_classes,
        feat_dim=feat_dim
    ) 
    network_opt = torch.optim.Adam(
        network.parameters(), lr=lr, weight_decay=weight_decay
    )
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        network_opt, step_size=lr_decay_every, gamma=lr_decay_by
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    network.to(device)
    optimizer_to_device(network_opt, device)

    print("Loading segmentation dataset ...... ")
    train_dataset = Dataset(
        os.path.join(data_dir, task), "train", train_num=train_num, val_num=val_num, data_file_name="data.npz"
    )
    val_dataset = Dataset(
        os.path.join(data_dir, task), "val", train_num=train_num, val_num=val_num, data_file_name="data.npz"
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8, 
        drop_last=True,
    )
    train_num_batch = len(train_dataloader)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        drop_last=False,
    )
    val_num_batch = len(val_dataloader)

    print(f"train_num_batch: {train_num_batch}, val_num_batch: {val_num_batch}")

    # start training
    start_epoch = 1
    print("Start training for Segmentation Task ...... ")

    for epoch in tqdm(
        range(start_epoch, num_epochs + 1),
        desc="Epoch",
        initial=start_epoch,
        total=num_epochs,
        leave=True,
    ):
        train_batches = enumerate(train_dataloader, 0)
        
        train_ep_loss, train_cnt = 0.0, 0
        train_ep_acc = 0.0
        
        network.train()
        for train_batch_ind, batch in tqdm(
            train_batches, desc="Batch", total=train_num_batch, leave=False
        ):
            
            total_loss, acc = forward_step(
                batch=batch, network=network, device=device, epoch=epoch, is_val=False
            )
            
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()

            train_ep_loss += total_loss.item()
            train_ep_acc += acc.item()
            train_cnt += 1
        
        network_lr_scheduler.step() 

        content = {
            "total_loss": train_ep_loss / train_cnt,
            "segmentation_acc": train_ep_acc / train_cnt,
            "lr": network_opt.param_groups[0]["lr"],
        }
        log_writer(epoch, content, is_val=False)

        if epoch % val_every_epoch == 0 and val_num_batch > 0:
            val_batches = enumerate(val_dataloader, 0)
            val_ep_loss, val_cnt = 0.0, 0
            val_ep_acc = 0.0
            
            network.eval()
            with torch.no_grad():
                for val_batch_ind, batch in tqdm(
                    val_batches, desc="Val", total=val_num_batch, leave=False
                ):
                    total_loss, acc = forward_step(
                        batch=batch,
                        network=network,
                        epoch=epoch,
                        device=device,
                        is_val=True,
                    )
                    val_ep_loss += total_loss.item()
                    val_ep_acc += acc.item()
                    val_cnt += 1
            
            content = {
                "loss": val_ep_loss / val_cnt,
                "segmentation_acc": val_ep_acc / val_cnt,
            }
            log_writer(epoch, content, is_val=True)

        if (
            epoch % save_every_epoch == 0
            or epoch == num_epochs
            or network_opt.param_groups[0]["lr"] < 5e-7
        ):
            with torch.no_grad():
                tqdm.write(f"Saving checkpoint {epoch}...... ")
                ckpt_dir = os.path.join(save_dir, "ckpts")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(
                    network.state_dict(),
                    os.path.join(ckpt_dir, f"{epoch}-network.pth"),
                )
                torch.save(
                    network_opt.state_dict(),
                    os.path.join(ckpt_dir, f"{epoch}-optimizer.pth"),
                )
                torch.save(
                    network_lr_scheduler.state_dict(),
                    os.path.join(ckpt_dir, f"{epoch}-lr_scheduler.pth"),
                )
                tqdm.write("DONE")
            if network_opt.param_groups[0]["lr"] < 5e-7:
                tqdm.write(
                    "Epoch %d : Learning rate is too small, stop training" % epoch
                )
                break


def get_acc(logits, labels):

    preds = torch.argmax(logits, dim=1)
    
    if labels.dtype != torch.long:
        labels = labels.long()
    correct = (preds == labels).float().sum()
    total_points = labels.shape[0] * labels.shape[1]
    accuracy = correct / total_points
    return accuracy


def forward_step(batch, network: Network, device, epoch=0, is_val=False):
    points, labels = batch
    pcs = points.to(device)     # (B, N, 6)
    labels = labels.to(device)  # (B, N)
    losses, logits = network.get_loss(pcs, labels)
    total_loss = losses["total_loss"]
    acc = get_acc(logits, labels)
    return total_loss, acc


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", type=str, default="final", help="task name")
    args.add_argument(
        "--train_num", type=int, default=7680, help="number of training data"
    )
    args.add_argument(
        "--val_num", type=int, default=1314, help="number of validation data"
    )
    args = args.parse_args()

    task = args.task

    config = {
        "val_every_epoch": 1,
        "save_every_epoch": 10,
        "train_num": args.train_num,
        "val_num": args.val_num,
        "point_cloud_dim": 3,  
        "num_classes": 7, 
        "feat_dim": 128,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "lr_decay_every": 500,
        "lr_decay_by": 0.9,
        "batch_size": 32, 
        "num_epochs": 1000,
    }
    data_dir = "gen_map/data"
    base_dir = "gen_map/logs"
    train(task, config, data_dir, base_dir)