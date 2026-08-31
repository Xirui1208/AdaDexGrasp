#!/bin/bash

cd $( dirname ${BASH_SOURCE[0]} )/../../../


python train.py \
    'task=TacSLTaskInsertion' 'task.env.task_type=insertion' \
    'seed=-1' 'task.env.use_gelsight=True' 'headless=False' \
    'max_iterations=1000' 'task.env.numEnvs=64' \
    'train=TacSLTaskInsertionPPO_LSTM_dict_AAC' 'train.params.config.horizon_length=512' \
    'train.params.config.mini_epochs=4' \
    +'task.env.obsDims={dof_pos:[24],dof_vel:[24],dof_force:[24],hand_base_pos:[3],hand_base_quat:[4],fingertip_state:[65],socket_pos:[3],socket_quat:[4]}' \
    +'train.params.network.input_preprocessors={dof_pos:{},dof_vel:{},dof_force:{},hand_base_pos:{},hand_base_quat:{},fingertip_state:{},socket_pos:{},socket_quat:{}}' \
    'task.rl.asymmetric_observations=True' 'task.rl.add_contact_info_to_aac_states=True' \
    +'task.env.stateDims={dof_pos:[24],dof_vel:[24],dof_force:[24],hand_base_pos:[3],hand_base_quat:[4],fingertip_state:[65],plug_pos:[3],plug_quat:[4],socket_pos_gt:[3],socket_quat:[4],plug_socket_force:[3],plug_palm_elastomer_force:[3],plug_thumb_elastomer_force:[3],plug_ff_elastomer_force:[3],plug_mf_elastomer_force:[3],plug_rf_elastomer_force:[3],plug_lf_elastomer_force:[3]}' \
    +'train.params.config.central_value_config.network.input_preprocessors={dof_pos:[24],dof_vel:[24],dof_force:[24],hand_base_pos:[3],hand_base_quat:[4],fingertip_state:[65],plug_pos:[3],plug_quat:[4],socket_pos_gt:[3],socket_quat:[4],plug_socket_force:[3],plug_palm_elastomer_force:[3],plug_thumb_elastomer_force:[3],plug_ff_elastomer_force:[3],plug_mf_elastomer_force:[3],plug_rf_elastomer_force:[3],plug_lf_elastomer_force:[3]}' \
    'task.rl.add_contact_force_plug_decomposed=True' \
    +'task.env.obsDims={plug_pos:[3],plug_quat:[4],plug_socket_force:[3],plug_palm_elastomer_force:[3],plug_thumb_elastomer_force:[3],plug_ff_elastomer_force:[3],plug_mf_elastomer_force:[3],plug_rf_elastomer_force:[3],plug_lf_elastomer_force:[3]}' \
    +'train.params.network.input_preprocessors={plug_pos:{},plug_quat:{},plug_socket_force:{},plug_palm_elastomer_force:{},plug_thumb_elastomer_force:{},plug_ff_elastomer_force:{},plug_mf_elastomer_force:{},plug_rf_elastomer_force:{},plug_lf_elastomer_force:{}}' \
    experiment=insert_full_state