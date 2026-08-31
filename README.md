# AdaDexGrasp: Adaptive Dexterous Grasping via 3D Visuo-Tactile Representation Fusion

<div align="center">

ECCV 2026

<a href="https://xirui1208.github.io/AdaDexGrasp/"><img src="https://img.shields.io/badge/Project-Website-356AE6.svg" alt="Project website"></a>
<a href="https://arxiv.org/abs/2608.07600"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B.svg" alt="Paper"></a>
<a href="https://arxiv.org/pdf/2608.07600"><img src="https://img.shields.io/badge/PDF-Download-4CAF50.svg" alt="PDF"></a>
<a href="https://github.com/Xirui1208/AdaDexGrasp"><img src="https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github" alt="Code"></a>

</div>

## Usage

### 1. Prepare UniDexGrasp++

Install and verify UniDexGrasp++, Isaac Gym, PointNet++, PyTorch3D, Open3D,
Hydra, diffusers, and Zarr 2.x. Prepare the UniDexGrasp++ assets and a compatible
vision-policy checkpoint.


### 2. Collect data

Configure the object in `dexgrasp/cfg/shadow_hand_random_load_vision.yaml` and
the policy path in `dexgrasp/script/run_collect.sh`.

Select the collector in `dexgrasp/utils/parse_task.py`:

```python
from dexgrasp.tasks.fail_colllection import ShadowHandRandomLoadVision
# or
from dexgrasp.tasks.success_collection import ShadowHandRandomLoadVision
```

Run both the success and failure collectors:

```bash
cd dexgrasp
PYTHONPATH=.. bash script/run_collect.sh
```


### 3. Train the success classifier

Save the merged data as
`dexgrasp/ECCV/classifier_data/classifier.npz`, then run:

```bash
cd dexgrasp
python classifier.py
```

This produces the classifier checkpoint and `rgb_stats.npz`.

### 4. Train the DP3 adjuster

Save the merged data as `dexgrasp/total.npz`, then run these scripts in order:

```bash
cd dexgrasp
python near.py

cd ../Diffusion_Policy_3D
python data2zarr_dp3.py ../dexgrasp/DATA/adapt-data.npz data/default_task
python train.py --config-name=robot_dp3.yaml
```

They perform pose pairing, Zarr conversion, and adjuster training respectively.


### 5. Train GenMap and GenPose

Set the successful-data path in `dexgrasp/seg_object.py`, then run:

```bash
cd dexgrasp
python seg_object.py
```

Copy the generated NPZ to:

```text
gen_map/data/<task>/data.npz
gen_pose/data/<task>/data.npz
```

Train both models from the repository root:

```bash
cd ..

python gen_map/train.py \
  --task <task> --train_num <train-size> --val_num <val-size>

python gen_pose/train.py \
  --task <task> --train_num <train-size> --val_num <val-size>
```


### 6. Run the final evaluation

Set the classifier, DP3, GenMap, and GenPose checkpoint paths at the bottom of
`dexgrasp/final.py`. Keep the matching `rgb_stats.npz` in `dexgrasp/` and
configure the object in `dexgrasp/cfg/final.yaml`.

```bash
cd dexgrasp
PYTHONPATH=.. python final.py
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{liang2026adadexgraspadaptivedexterousgrasping,
  title         = {AdaDexGrasp: Adaptive Dexterous Grasping via 3D Visuo-Tactile Representation Fusion},
  author        = {Xirui Liang and Jiaqi Liang and Jingkai Xu and Yuran Wang and Ruochong Li and Yuanpei Chen and Masayoshi Tomizuka and Wei Zhan and Ruihai Wu},
  year          = {2026},
  eprint        = {2608.07600},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2608.07600}
}
```
