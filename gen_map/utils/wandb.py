import wandb
import datetime
from tqdm import tqdm

debug = False


def log_init(run_name=None, cfg=None, mode="online"):

    if debug:
        mode = "disabled"
    _time = datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")
    run_name = f"{run_name}_{_time}"
    wandb.init(project="Grasp tactile", name=run_name, config=cfg, mode=mode)


def log_writer(epoch, content, is_val):
    prefix = "Val/" if is_val else "Train/"
    tqdm.write("|----------------------Epoch %d-------------------------|" % epoch)
    for key, value in content.items():
        tqdm.write("%s%s: %f" % (prefix, key, value))
        wandb.log({f"{prefix}{key}": value}, step=epoch)
    tqdm.write("|------------------------------------------------------|")
