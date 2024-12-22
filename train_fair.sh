#!/bin/bash
# specify the device id pytorch can see, eg. "0,1,2,3"
cuda_devices='0'
echo "Set environ CUDA devices $cuda_devices visible"
num_device=1
# specify the number of devices pytorch will use, the number must not be more than the number of devices initialized above
echo "Using $num_device CUDA devices for training"
# specify the sensitive attribute name
att_name='race'
# specify number of senstive classes
num_att=3
num_epochs=5
exp_name="TEST"
echo "Experiment name: $exp_name"

CUDA_VISIBLE_DEVICES=$cuda_devices torchrun --nproc_per_node=$num_device \
--rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 train_dro.py \
--task_name $exp_name \
--attribute_name $att_name \
--num_sensitive_classes $num_att \
--num_epochs $num_epochs \
--model_type "vit_b" \
--work_dir ./work_dir \
--checkpoint ./work_dir/SAM/sam_vit_b_01ec64.pth \
--batch_size 2 \
--num_workers 16 \
--bucket_cap_mb 25 \
--finetune_head \
--use_amp  > ./logs/${exp_name}.log
