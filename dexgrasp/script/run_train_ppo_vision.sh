CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py \
--task=ShadowHandRandomLoadVision \
--algo=ppo \
--seed=50 \
--rl_device=cuda:0 \
--sim_device=cuda:0 \
--num_finger_contact=2 \
--model_dir=newlog/usbstick_625e_010_seed0/model_10000.pt \
--vision \
--test \
--backbone_type pn #pn/transpn \
# --headless \
