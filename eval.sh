#!/bin/bash
# specify the device id pytorch can see, eg. "0,1,2,3"
cuda_devices="0"
echo "Set environ CUDA devices $cuda_devices visible"
CUDA_VISIBLE_DEVICES=$cuda_devices python evaluate_upgrade.py \
--model_type "vpt_vit_b" \
--checkpoint ./work_dir/GroupDRO-20241218-0738/medsam_model_best.pth \
--work_dir ./work_dir \
--batch_size 1 \
--num_workers 16
