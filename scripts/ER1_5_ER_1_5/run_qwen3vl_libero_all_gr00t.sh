#!/bin/bash

# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
date_time=$(date +%m%d_%H%M)
base_vlm=/home/tiger/.cache/huggingface/hub/models--Embodied1--Embodied-R1.5-SFT-v1/snapshots/0d2820fe1b7f598d2765b234ea69c351211b55c9
config_yaml=scripts/ER1_5/qwen3vl_libero.yaml
libero_data_root=./playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./Checkpoints
run_id=libero4in1_${Framework_name}_${date_time}
batch_size=8
wandb_project=Qwen3VL_libero_all_${Framework_name}
wandb_entity=lookas # set your wandb entity here
# === End of environment variable configuration ===

# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch --main_process_port 12773 \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${batch_size} \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project ${wandb_project} \
  --wandb_entity ${wandb_entity} \

##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
