# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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
from build_medsam import MedSAM, FairMedSAM
from ddp_utils import init_distributed_mode, gather_object_across_processes
from load_data import show_mask, show_box, NpzTrainSet, NpzTestSet


# set up parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ti", "--tr_npy_path", type=str, default="HarvardFairSeg/Training", help="path to training FairSeg dataset")
    parser.add_argument("-vi", "--val_npy_path", type=str, default="HarvardFairSeg/Test", help="path to testing FairSeg dataset")
    parser.add_argument("--task_name", type=str, default="MedSAM-finetune")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    # train
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    # finetuning ops
    parser.add_argument("--finetune_backbone", action='store_true', default=False, help='finetune the ViT backbone in SAM model')
    parser.add_argument("--finetune_mask_decoder", action='store_true', default=False, help='finetune the whole mask decoder in SAM model')
    parser.add_argument("--finetune_head", action='store_true', default=False, help='finetune the segmentation head of the mask decoder in SAM')
    # save ops
    parser.add_argument("--save_steps", type=int, default=500)
    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("--lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--use_wandb", type=bool, default=False, help="use wandb to monitor training")
    parser.add_argument("--use_amp", action="store_true", default=False, help="use amp")
    ## Distributed training args
    parser.add_argument("--world_size", type=int, default=1, help="world size")
    parser.add_argument("--bucket_cap_mb", type=int, default=25, 
                        help="The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)")
    parser.add_argument("--grad_acc_steps", type=int, default=1, help="Gradient accumulation steps before syncing gradients for backprop")
    parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
    parser.add_argument("--dist-url", type=str, default="env://")

    parser.add_argument("--num_groups", type=int, default=2, help="数据中的组数量")
    parser.add_argument("--is_robust", action='store_true', default=True, help="是否启用Group DRO")
    parser.add_argument("--step_size", type=float, default=0.01, help="更新对抗概率的步长")
    parser.add_argument("--alpha", type=float, default=1.0, help="Group DRO的平衡参数")

    args = parser.parse_args()
    return args

class LossComputer:
    def __init__(self, num_groups, is_robust, step_size, alpha):
        self.num_groups = num_groups
        self.is_robust = is_robust
        self.step_size = step_size
        self.alpha = alpha
        self.adv_probs = torch.ones(num_groups, device='cuda') / num_groups  # 初始化在GPU上

    def compute_group_loss(self, group_losses):
        epsilon = 1e-12
        adjusted_losses = group_losses + self.alpha * torch.log(self.adv_probs + epsilon)
        final_loss = torch.dot(adjusted_losses, self.adv_probs)
        with torch.no_grad():
            self.adv_probs = self.adv_probs * torch.exp(self.step_size * group_losses)
            self.adv_probs = self.adv_probs / self.adv_probs.sum()
        return final_loss

def main(args):
    args = parse_args()
    print(args)
    # set seeds for reproducibility
    torch.manual_seed(2023)
    torch.cuda.empty_cache()
    init_distributed_mode(args)

    if args.use_wandb:
        import wandb
        wandb.login()
        wandb.init(
            project=args.task_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "data_path": args.tr_npy_path,
                "model_type": args.model_type,
            },
        )

    # set up model for fine-tuning
    # device = args.device
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    os.makedirs(model_save_path, exist_ok=True)

    # build the SAM model
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder
        ).cuda()

    # setting up trainable parameters
    # let's first freeze all the parameters in SAM
    for param in sam_model.parameters():
        param.requires_grad = False
    # finetune backbone
    if args.finetune_backbone:
        for param in medsam_model.image_encoder.parameters():
            param.requires_grad = True
    
    # finetune mask decoder
    if args.finetune_mask_decoder:
        for param in medsam_model.mask_decoder.parameters():
            param.requires_grad = True
    
    # finetune the segmentation head
    if args.finetune_head:
        for param in medsam_model.mask_decoder.output_upscaling.parameters():
            param.requires_grad = True
        for mlp in medsam_model.mask_decoder.output_hypernetworks_mlps:
            for param in mlp.parameters():
                param.requires_grad = True
        for param in medsam_model.mask_decoder.iou_prediction_head.parameters():
            param.requires_grad = True
    
    # check if any visual prompts exist, if so, set these params to trainable
    for name, param in medsam_model.named_parameters():
        if 'visual_prompt_embeddings' in name:
            param.requires_grad = True
    
    num_trainable_parameters = sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)
    if num_trainable_parameters == 0:
        raise RuntimeError("No parameters to train. Please check the finetune options")
    # format output
    print('Number of trainable parameters: {}M'.format(num_trainable_parameters / 1e6))

    ## Setting up optimiser and loss func
    gpu = torch.device('cuda')
    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,  ## Too large -> comminitation overlap, too small -> unable to overlap with computation
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, medsam_model.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
    ce_loss = nn.BCEWithLogitsLoss(reduction="none")

    # define metrics
    dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="none")
    iou_metric = monai.metrics.MeanIoU(include_background=False, reduction="none")

    # train
    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10
    train_dataset = NpzTrainSet(args.tr_npy_path)
    test_set = NpzTestSet(args.val_npy_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)
    ## Distributed sampler has done the shuffling for you,
    ## So no need to shuffle in dataloader

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                ),
            )
        torch.distributed.barrier()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        # scaler = torch.amp.GradScaler()
        print("Using AMP for training")

    for epoch in range(start_epoch, num_epochs):
        sam_model.train()
        epoch_loss = 0
        train_dataloader.sampler.set_epoch(epoch)
        tbar = tqdm(range(len(train_dataloader)))
        train_iter = iter(train_dataloader)
        
        for step in tbar:
            data = next(train_iter)
            image = data['image'].cuda()
            gt2D = data['label'].cuda()
            boxes = data['bboxes'].detach().cpu().numpy()
            group_ids = data['group_id'].cuda()
            optimizer.zero_grad()

            medsam_pred = medsam_model(image, boxes)
            dice_loss_per_sample = seg_loss(medsam_pred, gt2D)
            ce_loss_per_sample = ce_loss(medsam_pred, gt2D.float())
            loss_per_sample = dice_loss_per_sample + ce_loss_per_sample

            # 计算每个组的损失
            group_losses = []
            for group_id in range(args.num_groups):
                group_mask = (group_ids == group_id)
                if group_mask.sum() == 0:
                    group_loss = torch.tensor(0.0, device=loss_per_sample.device)
                else:
                    group_loss = loss_per_sample[group_mask].mean()
                group_losses.append(group_loss)
            group_losses = torch.stack(group_losses)

            # 同步组损失到所有进程
            dist.all_reduce(group_losses, op=dist.ReduceOp.SUM)
            group_losses = group_losses / args.world_size

            if args.is_robust:
                # 使用LossComputer计算最终损失
                final_loss = loss_computer.compute_group_loss(group_losses)
            else:
                # 如果不使用Group DRO，直接取平均损失
                final_loss = loss_per_sample.mean()

            # 反向传播
            final_loss.backward()
            optimizer.step()

            epoch_loss += final_loss.item()
            running_loss = epoch_loss / (step + 1)
            tbar.set_postfix_str('Running Loss: {:.6f}'.format(running_loss))

            if step > 10 and step % args.save_steps == 0:
                if dist.get_rank() == 0:
                    checkpoint = {
                        "model": medsam_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(
                        checkpoint,
                        join(model_save_path, "medsam_model_latest_step.pth"),
                    )
                    print(f"Saved checkpoint to {join(model_save_path, 'medsam_model_latest_step.pth')}")

        losses.append(running_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": running_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {running_loss}'
        )
        # save the model checkpoint
        if dist.get_rank() == 0:
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_latest_epoch.pth"))
            print(f"Saved checkpoint to {join(model_save_path, 'medsam_model_latest_epoch.pth')}")
            ## save the best model
            if epoch_loss < best_loss:
                # # delete the previous best model
                # if os.path.exists(join(model_save_path, f"medsam_model_best_{best_loss:.4f}.pth")):
                #     os.remove(join(model_save_path, f"medsam_model_best_{best_loss:.4f}.pth"))
                # print(f"Deleted checkpoint to {join(model_save_path, f'medsam_model_best_{best_loss:.4f}.pth')}")
                best_loss = epoch_loss
                print(f"Best loss: {best_loss}")
                torch.save(checkpoint, join(model_save_path, f"medsam_model_best_{best_loss:.4f}_epoch_{epoch}.pth"))
                print(f"Saved checkpoint to {join(model_save_path, f'medsam_model_best_{best_loss:.4f}_epoch_{epoch}.pth')}")
        torch.distributed.barrier()

        # plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.show() # comment this line if you are running on a server
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()

        # run the evaluation loop
        medsam_model.eval()
        dice_metric.reset()
        iou_metric.reset()
        tbar = tqdm(range(len(test_dataloader)))
        test_iter = iter(test_dataloader)
        metric_list = []
        for step in tbar:
            data = next(test_iter)
            image = data['image']
            gt_cup = data['cup_label']
            gt_disc = data['disc_label']
            gt_rim = torch.where(gt_disc - gt_cup > 0, 1, 0)
            cup_bboxes = data['cup_bboxes']
            disc_bboxes = data['disc_bboxes']
            cup_boxes_np = cup_bboxes.detach().cpu().numpy()
            disc_boxes_np = disc_bboxes.detach().cpu().numpy()
            image = image.cuda()
            with torch.no_grad(), torch.amp.autocast('cuda'):
                cup_pred = (torch.sigmoid(medsam_model(image, cup_boxes_np)) >= 0.5).to(torch.float32).cpu()
                disc_pred = (torch.sigmoid(medsam_model(image, disc_boxes_np)) >= 0.5).to(torch.float32).cpu()
                rim_pred = torch.where(disc_pred - cup_pred > 0, 1, 0)
                cup_dice = dice_metric(cup_pred, gt_cup)
                rim_dice = dice_metric(rim_pred, gt_rim)
                cup_iou = iou_metric(cup_pred, gt_cup)
                rim_iou = iou_metric(rim_pred, gt_rim)
                for idx in range(args.batch_size):
                    name = data['img_name'][idx]
                    cup_dice_val = cup_dice[idx].item()
                    rim_dice_val = rim_dice[idx].item()
                    cup_iou_val = cup_iou[idx].item()
                    rim_iou_val = rim_iou[idx].item()
                    metric_dict = {'cup_dice': cup_dice_val, 'rim_dice': rim_dice_val, 'cup_iou': cup_iou_val, 'rim_iou': rim_iou_val}
                    metric_sublist = gather_object_across_processes(metric_dict)
                    metric_list.extend(metric_sublist)
        metric_df = pd.DataFrame(metric_list)
        stats = ''
        # iter the column names in metric_df and report their average
        for col in metric_df.columns:
            stats += f'{col}: {metric_df[col].mean():.4f} '
        print(stats)

    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    main(args)
