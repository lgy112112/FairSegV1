#!/bin/bash
# specify the device id pytorch can see, eg. "0,1,2,3"
cuda_devices="0"
echo "Set environ CUDA devices $cuda_devices visible"
num_device=1
# specify the number of devices pytorch will use, the number must not be more than the number of devices initialized above
echo "Using $num_device CUDA devices for training"
exp_name="medsam-vit_b-$((num_device))gpus"
echo "Experiment name: $exp_name"
CUDA_VISIBLE_DEVICES=$cuda_devices torchrun --nproc_per_node=$num_device \
--rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 train_medsam.py \
--task_name $exp_name \
--num_epochs 5 \
--model_type "vit_b" \
--work_dir ./work_dir \
--checkpoint ./work_dir/SAM/sam_vit_b_01ec64.pth \
--batch_size 4 \
--bucket_cap_mb 25 \
--grad_acc_steps 1 \
--finetune_head \
--use_amp  > ./logs/log_for_${exp_name}.log