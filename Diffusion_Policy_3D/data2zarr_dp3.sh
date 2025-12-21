# Examples:
# bash data2zarr_dp3.sh Hang_Tops 1 100

# 'task_name' e.g.Fold_Dress, Hang_Tops, Fling_Trousers, etc.
# 'stage_index' e.g. 1, 2, 3, etc.
# 'train_data_num' means number of training data, e.g. 100, 200, 300, etc.



input_data_path=${1}
output_data_path=${2}
python_path='/home/isaac/anaconda3/envs/dexgrasp/bin/python3.8'

export PYTHON_PATH=$python_path

$python_path data2zarr_dp3.py ${input_data_path} ${output_data_path} 
