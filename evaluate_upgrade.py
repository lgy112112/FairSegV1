# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from build_medsam import MedSAM
from ddp_utils import init_distributed_mode
from load_data import show_mask, show_box, NpzTestSet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()


# %% set up parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--val_npy_path", type=str, default="HarvardFairSeg/Test", help="path to testing FairSeg dataset")
    parser.add_argument("--savedir", type=str, default=None, help="directory to saved models")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
    # val
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args()

    # 直接加载指定的权重路径
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        # 修复分布式训练的权重文件（如果需要）
        original_ckpt = torch.load(args.checkpoint, map_location='cpu')['model']
        new_ckpt = {key.replace('module.', ''): value for key, value in original_ckpt.items()}
        fixed_ckpt_path = args.checkpoint.replace(".pth", "_fixed.pth")
        torch.save(new_ckpt, fixed_ckpt_path)

        # 加载修复后的权重到 SAM 模型
        sam_model = sam_model_registry[args.model_type](checkpoint=fixed_ckpt_path)
    else:
        raise ValueError("Checkpoint path must be provided using --checkpoint")

    # 构建 MedSAM 模型
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder
    ).to(device)
    medsam_model.eval()

    # 加载测试数据
    test_set = NpzTestSet(args.val_npy_path)
    print("Number of testing samples: ", len(test_set))
    test_dataloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 初始化评估指标
    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="none")
    iou_metric = monai.metrics.MeanIoU(include_background=False, reduction="none")

    # 开始评估
    tbar = tqdm(range(len(test_dataloader)))
    test_iter = iter(test_dataloader)
    metric_list = []
    for step in tbar:
        data = next(test_iter)
        image = data['image'].to(device)
        gt_cup = data['cup_label']
        gt_disc = data['disc_label']
        gt_rim = torch.where(gt_disc - gt_cup > 0, 1, 0)  # 计算 Rim 标签
        cup_bboxes = data['cup_bboxes']
        disc_bboxes = data['disc_bboxes']
        cup_boxes_np = cup_bboxes.detach().cpu().numpy()
        disc_boxes_np = disc_bboxes.detach().cpu().numpy()

        # 使用 MedSAM 模型预测
        with torch.no_grad(), torch.amp.autocast('cuda'):
            cup_pred = (torch.sigmoid(medsam_model(image, cup_boxes_np)) >= 0.5).to(torch.float32).cpu()
            disc_pred = (torch.sigmoid(medsam_model(image, disc_boxes_np)) >= 0.5).to(torch.float32).cpu()
            rim_pred = torch.where(disc_pred - cup_pred > 0, 1, 0)

        # 计算 Dice 和 IoU
        cup_dice = dice_metric(cup_pred, gt_cup)
        rim_dice = dice_metric(rim_pred, gt_rim)
        cup_iou = iou_metric(cup_pred, gt_cup)
        rim_iou = iou_metric(rim_pred, gt_rim)

        # 保存每个样本的结果
        for idx in range(args.batch_size):
            name = data['img_name'][idx]
            metric_list.append({
                'name': name,
                'cup_dice': cup_dice[idx].item(),
                'rim_dice': rim_dice[idx].item(),
                'cup_iou': cup_iou[idx].item(),
                'rim_iou': rim_iou[idx].item()
            })

    # 保存评估结果
    metric_df = pd.DataFrame(metric_list)
    metric_df.to_csv(join(args.work_dir, f'vit_metric.csv'), index=False)


    

            
if __name__ == "__main__":
    args = parse_args()
    main(args)
