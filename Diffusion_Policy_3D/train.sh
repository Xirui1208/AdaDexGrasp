# Examples:
# bash train.sh Hang_Tops_stage_1 100 42 0 False

# 'task_name' must be {task}_stage_{stage_index}, e.g. Hang_Tops_stage_1
# 'expert_data_num' means number of training data. e.g. 100
# 'seed' means random seed, select any number you like, e.g. 42
# 'gpu_id' means single gpu id, e.g.0
# 'DEBUG' means whether to run in debug mode. e.g. False. 
# Before Run, you can set 'DEBUG'=True to check if the code is running correctly.

task_name=${1}
data_dir=${2}
seed=${3}
gpu_id=${4}
DEBUG=${5} # True or False

save_ckpt=True

python_path='/home/isaac/anaconda3/envs/dexgrasp/bin/python3.8'

export PYTHON_PATH=$python_path

alg_name=robot_dp3
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="info/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

$PYTHON_PATH train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path=${data_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}
