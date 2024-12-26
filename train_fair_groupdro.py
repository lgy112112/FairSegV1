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
from build_medsam import MedSAM, FairMedSAM, Predictor, ShannonEntropy
from ddp_utils import init_distributed_mode, gather_object_across_processes
from load_data import show_mask, show_box, NpzTrainSet, NpzTestSet

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()


# %% sanity test of dataset class
tr_dataset = NpzTrainSet("./HarvardFairSeg/Training")
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
for step, data in enumerate(tr_dataloader):
    image = data['image']
    gt = data['label']
    bboxes = data['bboxes']
    names_temp = data['img_name']
    # print(image.shape, gt.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, 7)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    # set title
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    # set title
    axs[1].set_title(names_temp[idx])
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break

# %% set up parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ti", "--tr_npy_path", type=str, default="HarvardFairSeg/Training", help="path to training FairSeg dataset")
    parser.add_argument("-vi", "--val_npy_path", type=str, default="HarvardFairSeg/Test", help="path to testing FairSeg dataset")
    parser.add_argument("--task_name", type=str, default="MedSAM-finetune")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--attribute_name", type=str, required=True, help="name of the sensitive attribute to predict")
    parser.add_argument("--num_sensitive_classes", type=int, required=True, help="number of sensitive classes")
    # train
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0, help="weight for the sensitive prediction loss")
    parser.add_argument("--beta", type=float, default=0.1, help="weight for the entropy loss")
    parser.add_argument("--step_size", type=float, default=0.01, help="更新对抗概率的步长")
    # finetuning ops
    parser.add_argument("--finetune_backbone", action='store_true', default=False, help='finetune the ViT backbone in SAM model')
    parser.add_argument("--finetune_mask_decoder", action='store_true', default=False, help='finetune the whole mask decoder in SAM model')
    parser.add_argument("--finetune_head", action='store_true', default=False, help='finetune the segmentation head of the mask decoder in SAM')
    # save ops
    parser.add_argument("--save_steps", type=int, default=1000)
    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
    parser.add_argument("--lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--use_wandb", type=bool, default=False, help="use wandb to monitor training")
    parser.add_argument("--use_amp", action="store_true", default=False, help="use amp")
    ## Distributed training args
    parser.add_argument("--world_size", type=int, default=1, help="world size")
    parser.add_argument("--bucket_cap_mb", type=int, default=25, 
                        help="The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)")
    parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
    parser.add_argument("--dist-url", type=str, default="env://")

    args = parser.parse_args()
    return args

class LossComputer:
    def __init__(self, num_groups, is_robust, step_size, alpha, device='cuda'):
        self.num_groups = num_groups  # 组的数量
        self.is_robust = is_robust  # 是否启用Group DRO
        self.step_size = step_size  # 动态调整的步长
        self.alpha = alpha  # 调整参数
        self.device = device  # 设备
        self.adv_probs = torch.ones(num_groups, device=device) / num_groups  # 初始化每个组的权重（均匀分布）

    def compute_group_loss(self, group_losses):
        # 确保adv_probs在与group_losses相同的设备上
        self.adv_probs = self.adv_probs.to(group_losses.device)
        # 调整每个组的损失
        adjusted_losses = group_losses + self.alpha * torch.log(self.adv_probs)
        # 计算最终的损失（加权求和）
        final_loss = torch.dot(adjusted_losses, self.adv_probs)
        # 动态更新每个组的权重
        with torch.no_grad():
            self.adv_probs = self.adv_probs * torch.exp(self.step_size * group_losses)
            self.adv_probs = self.adv_probs / self.adv_probs.sum()  # 归一化
        return final_loss

def main(args):
    args = parse_args()
    init_distributed_mode(args)
    linear_layer = nn.Linear(768, args.num_sensitive_classes).cuda()
    vpt_linear_layer = nn.Linear(768, args.num_sensitive_classes).cuda()
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
    medsam_model = FairMedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        num_sensitive_classes=args.num_sensitive_classes,
    ).cuda().train()
    # setting up trainable parameters
    # let's first freeze all the parameters in SAM
    for param in sam_model.parameters():
        param.requires_grad = False
    
    # first unfreeze all parameters of the sensitive predictor
    # for param in medsam_model.sensitive_predictor.parameters():
    #     param.requires_grad = True

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
    
    num_all_parameters = sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)
    # num_predictor_parameters = sum(p.numel() for p in medsam_model.sensitive_predictor.parameters() if p.requires_grad)
    # num_trainable_parameters = num_all_parameters - num_predictor_parameters
    num_trainable_parameters = num_all_parameters

    if num_trainable_parameters == 0:
        raise RuntimeError("No parameters to train. Please check the finetune options")
    # format output
    print('Number of trainable parameters: {}M'.format(num_trainable_parameters / 1e6))
    
    gpu = torch.device('cuda')
    medsam_model = nn.parallel.DistributedDataParallel(
        medsam_model,
        device_ids=[gpu],
        output_device=gpu,
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
        bucket_cap_mb=args.bucket_cap_mb,  ## Too large -> comminitation overlap, too small -> unable to overlap with computation
    )

    ## Setting up optimiser and loss func
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, medsam_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    # loss func for segmentation
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # loss func for sensitive attribute prediction
    ce_loss_sensitive = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)
    entropy = ShannonEntropy(use_softmax=True)

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
    # Print the number of trainable parameters
    trainable_parameters = sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_parameters / 1e6:.4f}M")

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
        print("Using AMP for training")

    # 初始化LossComputer
    loss_computer = LossComputer(
        num_groups=args.num_sensitive_classes,  # 敏感类别的数量
        is_robust=True,  # 启用Group DRO
        step_size=args.step_size,  # 动态调整的步长
        alpha=args.alpha,  # 调整参数
        device='cuda',  # 设备
    )

    for epoch in range(start_epoch, num_epochs):
        sam_model.train()
        epoch_sam_loss = 0
        train_dataloader.sampler.set_epoch(epoch)
        tbar = tqdm(range(len(train_dataloader)))
        train_iter = iter(train_dataloader)
        for step in tbar:
            # 获取数据
            data = next(train_iter)
            image = data['image'].cuda()
            gt2D = data['label'].cuda()
            boxes = data['bboxes']
            sensitive_cls = data[args.attribute_name].cuda()  # 敏感类别标签
            boxes_np = boxes.detach().cpu().numpy()

            # 清零梯度
            optimizer.zero_grad()

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # 前向传播
                    medsam_pred, img_pred, vpt_pred = medsam_model(image, boxes_np)

                    # 计算每个组的损失
                    group_losses = []
                    for group_id in range(args.num_sensitive_classes):
                        group_mask = (sensitive_cls == group_id)  # 当前组的掩码
                        if not group_mask.any():
                            group_sam_loss = torch.tensor(0.0, device=image.device)
                        else:
                            group_medsam_pred = medsam_pred[group_mask]
                            group_gt2D = gt2D[group_mask]
                            group_sam_loss = seg_loss(group_medsam_pred, group_gt2D) + ce_loss(group_medsam_pred, group_gt2D.float())
                        group_losses.append(group_sam_loss)
                    # 将group_losses转换为张量
                    group_losses = torch.stack(group_losses)
                    # 在分布式训练中聚合所有进程的组损失
                    torch.distributed.all_reduce(group_losses)
                    # 使用LossComputer计算最终损失
                    final_loss = loss_computer.compute_group_loss(group_losses)
                    # 反向传播
                    scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                medsam_pred, img_pred, vpt_pred = medsam_model(image, boxes_np)

                # 计算每个组的损失
                group_losses = []
                for group_id in range(args.num_sensitive_classes):
                    group_mask = (sensitive_cls == group_id)
                    if not group_mask.any():
                        group_loss = torch.tensor(0.0, device=image.device)
                    else:
                        group_medsam_pred = medsam_pred[group_mask]
                        group_gt2D = gt2D[group_mask]
                        group_sam_loss = seg_loss(group_medsam_pred, group_gt2D) + ce_loss(group_medsam_pred, group_gt2D.float())
                    group_losses.append(group_sam_loss)
                # 将group_losses转换为张量
                group_losses = torch.stack(group_losses)
                # 在分布式训练中聚合所有进程的组损失
                torch.distributed.all_reduce(group_losses)
                # 使用LossComputer计算最终损失
                final_loss = loss_computer.compute_group_loss(group_losses)
                # 反向传播
                final_loss.backward()
                optimizer.step()

            # 累积损失
            epoch_sam_loss += final_loss.item()
            # 更新进度条
            running_sam_loss = epoch_sam_loss / (step + 1)
            stats = f'Epoch: {epoch}, Seg Loss: {running_sam_loss:.6f}'
            tbar.set_postfix_str(stats)

            # 每隔一定步数保存模型
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
        # 每个epoch结束时保存模型
        if dist.get_rank() == 0:
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, f"medsam_model_{epoch}.pth"))
            # 保存最佳模型
            if epoch_sam_loss < best_loss:
                best_loss = epoch_sam_loss
                torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
        torch.distributed.barrier()
        # %% plot loss
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
                cup_pred = (torch.sigmoid(medsam_model(image, cup_boxes_np, False)) >= 0.5).to(torch.float32).cpu()
                disc_pred = (torch.sigmoid(medsam_model(image, disc_boxes_np, False)) >= 0.5).to(torch.float32).cpu()
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
