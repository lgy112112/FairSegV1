#!/bin/bash
# specify the device id pytorch can see, eg. "0,1,2,3"
cuda_devices="0"
echo "Set environ CUDA devices $cuda_devices visible"
CUDA_VISIBLE_DEVICES=$cuda_devices python evaluate_upgrade.py \
--model_type "vit_b" \
--checkpoint ./work_dir/medsam-vit_b-1gpus-20241121-1557/medsam_model_latest_epoch.pth \
--work_dir ./work_dir \
--batch_size 8 \
--num_workers 16
