import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
import yaml
import time
import shutil
import random
import logging
import pprint
import torch
import sys
import argparse
import gc
import numpy as np
from einops import rearrange
from torch.optim import SGD, AdamW
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.efficient import ACDCsemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import count_params, init_log, AverageMeter, DiceLoss
from utils.val import eval_2d


def get_parser(datasetname):
    cfgs = DATASET_CONFIGS[datasetname]
    parser = argparse.ArgumentParser(description=datasetname)
    parser.add_argument('--dataset', type=str, default=datasetname, choices=DATASET_CONFIGS.keys())
    parser.add_argument('--base_dir', type=str, default=cfgs['base_dir'])
    parser.add_argument('--labelnum', type=int, default=cfgs['labelnum'], help=cfgs.get('label_help'))
    parser.add_argument('--num', default=cfgs.get('num'), type=int, help='unlabeled data number')
    parser.add_argument('--config', type=str, default=cfgs['config'], help='Path to config file.')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=str, default=False)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'])
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    parser.add_argument('--rampup_divisor', type=int, default=150)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--top_num', type=int, default=4)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--val_start', type=float, default=0.3)
    return parser


def init_2d_weight(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def build_class_names(cfg, args):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(args.dataset) or CLASSES.get(args.dataset.upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def build_dataloaders(args, cfg):
    trainset_u = ACDCsemiDataset('train_u', args, cfg['crop_size'])
    trainset_l = ACDCsemiDataset('train_l', args, cfg['crop_size'])
    valset = ACDCsemiDataset('val', args, cfg['crop_size'])
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    labeled_bs = cfg['batch_size']
    unlabeled_bs = cfg['batch_size']
    total_batch_size = labeled_bs + unlabeled_bs
    train_dataset = ConcatDataset([trainset_l, trainset_u])
    labeled_idxs = list(range(0, len(trainset_l)))
    unlabeled_idxs = list(range(len(trainset_l), len(trainset_l) + len(trainset_u)))
    batch_sampler = TwoStreamBatchSampler(
        primary_indices=unlabeled_idxs,
        secondary_indices=labeled_idxs,
        batch_size=total_batch_size,
        secondary_batch_size=labeled_bs,
    )
    trainloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=6,
        pin_memory=True,
        collate_fn=mix_collate_fn,
    )
    return trainloader, valloader


def build_optimizer(model, args, cfg):
    if args.optimizer == 'AdamW':
        return AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    return SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001, nesterov=True)


def sigmoid_rampup(current, rampup_length):
    if rampup_length <= 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(iter_num, args):
    return args.consistency * sigmoid_rampup(iter_num / max(args.rampup_divisor, 1), args.consistency_rampup)


def derive_patch_grid(cfg, args):
    crop_size = cfg.get('crop_size')
    if isinstance(crop_size, int):
        height, width = crop_size, crop_size
    elif isinstance(crop_size, (list, tuple)):
        if len(crop_size) == 1:
            height = width = int(crop_size[0])
        else:
            height, width = int(crop_size[-2]), int(crop_size[-1])
    else:
        raise ValueError('crop_size must be int or sequence')
    if height % args.patch_size != 0 or width % args.patch_size != 0:
        raise ValueError(f'crop_size {crop_size} must be divisible by patch_size {args.patch_size}')
    return height // args.patch_size, width // args.patch_size


def strong_augment_labeled(img):
    scale = torch.empty(img.shape[0], 1, 1, 1, device=img.device).uniform_(0.9, 1.1)
    bias = torch.empty(img.shape[0], 1, 1, 1, device=img.device).uniform_(-0.05, 0.05)
    noise = torch.randn_like(img) * 0.03
    return img * scale + bias + noise


def masked_dice_loss_from_logits(logits, target, valid_mask, nclass):
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    valid_mask = valid_mask.bool()
    loss = 0.0
    valid_count = 0
    for cls_idx in range(nclass):
        pred_cls = (pred == cls_idx) & valid_mask
        target_cls = (target == cls_idx) & valid_mask
        denom = pred_cls.sum().float() + target_cls.sum().float()
        if denom.item() == 0:
            continue
        inter = (pred_cls & target_cls).sum().float()
        loss += 1.0 - (2.0 * inter + 1e-10) / (denom + 1e-10)
        valid_count += 1
    if valid_count == 0:
        return logits.sum() * 0.0
    return loss / valid_count


def abd_i(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong, args, h_size, w_size, num_lb):
    image_patch_supervised_1 = rearrange(volume_batch[:num_lb].clone(), 'b 1 (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    image_patch_supervised_2 = rearrange(volume_batch_strong[:num_lb].clone(), 'b 1 (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_1 = rearrange(label_batch[:num_lb].clone(), 'b (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised_2 = rearrange(label_batch_strong[:num_lb].clone(), 'b (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    patches_supervised_1 = rearrange(outputs1_max[:num_lb], 'b (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    patches_supervised_2 = rearrange(outputs2_max[:num_lb], 'b (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    patches_mean_supervised_1 = patches_supervised_1.float().mean(dim=2)
    patches_mean_supervised_2 = patches_supervised_2.float().mean(dim=2)
    e = torch.argmax(patches_mean_supervised_1, dim=1)
    f = torch.argmin(patches_mean_supervised_1, dim=1)
    g = torch.argmax(patches_mean_supervised_2, dim=1)
    h = torch.argmin(patches_mean_supervised_2, dim=1)
    for idx in range(num_lb):
        if random.random() < 0.5:
            image_patch_supervised_1[idx, e[idx]] = image_patch_supervised_2[idx, h[idx]]
            image_patch_supervised_2[idx, g[idx]] = image_patch_supervised_1[idx, f[idx]]
            label_patch_supervised_1[idx, e[idx]] = label_patch_supervised_2[idx, h[idx]]
            label_patch_supervised_2[idx, g[idx]] = label_patch_supervised_1[idx, f[idx]]
    image_patch_supervised = torch.cat([image_patch_supervised_1, image_patch_supervised_2], dim=0)
    label_patch_supervised = torch.cat([label_patch_supervised_1, label_patch_supervised_2], dim=0)
    image_patch_supervised = rearrange(image_patch_supervised, 'b (h w) (p1 p2) -> b 1 (h p1) (w p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    label_patch_supervised = rearrange(label_patch_supervised, 'b (h w) (p1 p2) -> b (h p1) (w p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    return image_patch_supervised, label_patch_supervised


def abd_r(outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel, args, h_size, w_size, num_lb, num_ulb):
    image_patch_1 = rearrange(volume_batch[num_lb:].clone(), 'b 1 (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    image_patch_2 = rearrange(volume_batch_strong[num_lb:].clone(), 'b 1 (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    patches_1 = rearrange(outputs1_max[num_lb:], 'b (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    patches_2 = rearrange(outputs2_max[num_lb:], 'b (h p1) (w p2) -> b (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    patches_mean_1 = patches_1.float().mean(dim=2)
    patches_mean_2 = patches_2.float().mean(dim=2)
    patches_outputs_1 = rearrange(outputs1_unlabel, 'b c (h p1) (w p2) -> b c (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    patches_outputs_2 = rearrange(outputs2_unlabel, 'b c (h p1) (w p2) -> b c (h w) (p1 p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)
    patches_mean_outputs_1 = patches_outputs_1.detach().mean(dim=3).permute(0, 2, 1)
    patches_mean_outputs_2 = patches_outputs_2.detach().mean(dim=3).permute(0, 2, 1)
    top_num = max(1, min(args.top_num, patches_mean_1.shape[1]))
    top1_idx = torch.topk(patches_mean_1, top_num, dim=1).indices
    top2_idx = torch.topk(patches_mean_2, top_num, dim=1).indices
    for idx in range(num_ulb):
        kl_similarities_1 = torch.empty(top_num, device=volume_batch.device)
        kl_similarities_2 = torch.empty(top_num, device=volume_batch.device)
        b_idx = torch.argmin(patches_mean_1[idx], dim=0)
        d_idx = torch.argmin(patches_mean_2[idx], dim=0)
        patches_mean_outputs_min_1 = patches_mean_outputs_1[idx, b_idx]
        patches_mean_outputs_min_2 = patches_mean_outputs_2[idx, d_idx]
        patches_mean_outputs_top_1 = patches_mean_outputs_1[idx, top1_idx[idx]]
        patches_mean_outputs_top_2 = patches_mean_outputs_2[idx, top2_idx[idx]]
        for top_idx in range(top_num):
            kl_similarities_1[top_idx] = torch.nn.functional.kl_div(
                patches_mean_outputs_top_1[top_idx].softmax(dim=-1).log(),
                patches_mean_outputs_min_2.softmax(dim=-1),
                reduction='sum',
            )
            kl_similarities_2[top_idx] = torch.nn.functional.kl_div(
                patches_mean_outputs_top_2[top_idx].softmax(dim=-1).log(),
                patches_mean_outputs_min_1.softmax(dim=-1),
                reduction='sum',
            )
        a_idx = top1_idx[idx, torch.argmin(kl_similarities_1)]
        c_idx = top2_idx[idx, torch.argmin(kl_similarities_2)]
        image_patch_1[idx, b_idx] = image_patch_2[idx, c_idx]
        image_patch_2[idx, d_idx] = image_patch_1[idx, a_idx]
    image_patch = torch.cat([image_patch_1, image_patch_2], dim=0)
    return rearrange(image_patch, 'b (h w) (p1 p2) -> b 1 (h p1) (w p2)', h=h_size, w=w_size, p1=args.patch_size, p2=args.patch_size)


def evaluate_model(model, valloader, cfg, class_names, val_mode, writer, epoch, logger):
    model.eval()
    mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode=val_mode)
    for cls_idx, dice in enumerate(dice_class):
        class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
        logger.info('*** Evaluation [{}]: Class [{:} {:}] Dice: {:.3f}'.format(val_mode, cls_idx + 1, class_name, dice.item()))
        writer.add_scalar(f'eval/{val_mode}_{class_name}_DICE', dice.item(), epoch)
    logger.info('*** Evaluation [{}] MeanDice: {:.3f}'.format(val_mode, mDice.item()))
    writer.add_scalar(f'eval/{val_mode}_mDice', mDice.item(), epoch)
    return mDice


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model1 = init_2d_weight(UNet(in_chns=1, class_num=cfg['nclass']).cuda()).cuda()
    model2 = init_2d_weight(UNet(in_chns=1, class_num=cfg['nclass']).cuda()).cuda()
    optimizer1 = build_optimizer(model1, args, cfg)
    optimizer2 = build_optimizer(model2, args, cfg)
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(count_params(model1) + count_params(model2)))

    trainloader, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d' % total_iters)
    class_names = build_class_names(cfg, args)
    h_size, w_size = derive_patch_grid(cfg, args)

    pre_best_dice1 = 0.0
    pre_best_dice2 = 0.0
    best_epoch1 = 0
    best_epoch2 = 0
    epoch = -1
    iter_num = 0
    start_time = time.time()
    latest_path = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model1.load_state_dict(checkpoint['model1'])
        model2.load_state_dict(checkpoint['model2'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        epoch = checkpoint['epoch']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        pre_best_dice1 = checkpoint['previous_best1']
        pre_best_dice2 = checkpoint['previous_best2']
        best_epoch1 = checkpoint['best_epoch1']
        best_epoch2 = checkpoint['best_epoch2']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    log_interval = max(len(trainloader) // 8, 1)
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum:{args.labelnum}, '
            f'Previous best mdice model1: {pre_best_dice1:.4f} @epoch:{best_epoch1}, '
            f'model2: {pre_best_dice2:.4f} @epoch:{best_epoch2}'
        )
        total_loss = AverageMeter()
        total_loss_m1 = AverageMeter()
        total_loss_m2 = AverageMeter()
        total_consistency = AverageMeter()
        total_mask_ratio = AverageMeter()

        model1.train()
        model2.train()
        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_u_w, img_u_s, ignore_mask, _ = unlabeled_data
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)
            img_u_w = img_u_w.cuda(non_blocking=True)
            img_u_s = img_u_s.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)

            img_x_s = strong_augment_labeled(img_x)
            mask_x_s = mask_x.clone()
            volume_batch = torch.cat((img_x, img_u_w), dim=0)
            volume_batch_strong = torch.cat((img_x_s, img_u_s), dim=0)
            label_batch = torch.cat((mask_x, torch.zeros_like(ignore_mask)), dim=0)
            label_batch_strong = torch.cat((mask_x_s, torch.zeros_like(ignore_mask)), dim=0)
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            outputs1 = model1(volume_batch)
            outputs2 = model2(volume_batch_strong)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            outputs1_max = outputs_soft1.detach().max(dim=1)[0]
            outputs2_max = outputs_soft2.detach().max(dim=1)[0]
            outputs1_unlabel = outputs1[num_lb:]
            outputs2_unlabel = outputs2[num_lb:]
            pseudo_outputs1 = torch.argmax(outputs_soft1[num_lb:].detach(), dim=1)
            pseudo_outputs2 = torch.argmax(outputs_soft2[num_lb:].detach(), dim=1)

            image_patch_supervised, label_patch_supervised = abd_i(
                outputs1_max, outputs2_max, volume_batch, volume_batch_strong, label_batch, label_batch_strong,
                args, h_size, w_size, num_lb
            )
            image_output_supervised_1 = model1(image_patch_supervised)
            image_output_supervised_2 = model2(image_patch_supervised)

            image_patch_last = abd_r(
                outputs1_max, outputs2_max, volume_batch, volume_batch_strong, outputs1_unlabel, outputs2_unlabel,
                args, h_size, w_size, num_lb, num_ulb
            )
            image_output_1 = model1(image_patch_last)
            image_output_2 = model2(image_patch_last)
            pseudo_image_output_1 = torch.argmax(torch.softmax(image_output_1.detach(), dim=1), dim=1)
            pseudo_image_output_2 = torch.argmax(torch.softmax(image_output_2.detach(), dim=1), dim=1)

            loss1 = 0.5 * (criterion_l(outputs1[:num_lb], mask_x.long()) + diceloss(outputs1[:num_lb], mask_x))
            loss2 = 0.5 * (criterion_l(outputs2[:num_lb], mask_x_s.long()) + diceloss(outputs2[:num_lb], mask_x_s))
            valid_mask_u = ignore_mask != 255
            pseudo_supervision1 = masked_dice_loss_from_logits(outputs1_unlabel, pseudo_outputs2, valid_mask_u, cfg['nclass'])
            pseudo_supervision2 = masked_dice_loss_from_logits(outputs2_unlabel, pseudo_outputs1, valid_mask_u, cfg['nclass'])

            displaced_target = label_patch_supervised.long()
            loss3 = 0.5 * (criterion_l(image_output_supervised_1, displaced_target) + diceloss(image_output_supervised_1, displaced_target))
            loss4 = 0.5 * (criterion_l(image_output_supervised_2, displaced_target) + diceloss(image_output_supervised_2, displaced_target))

            valid_patch_mask = torch.ones_like(pseudo_image_output_1, dtype=torch.bool)
            pseudo_supervision3 = masked_dice_loss_from_logits(image_output_1, pseudo_image_output_2, valid_patch_mask, cfg['nclass'])
            pseudo_supervision4 = masked_dice_loss_from_logits(image_output_2, pseudo_image_output_1, valid_patch_mask, cfg['nclass'])

            consistency_weight = get_current_consistency_weight(iter_num, args)
            model1_loss = loss1 + loss3 + consistency_weight * (pseudo_supervision1 + pseudo_supervision3)
            model2_loss = loss2 + loss4 + consistency_weight * (pseudo_supervision2 + pseudo_supervision4)
            loss = model1_loss + model2_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            current_iter = iter_num
            lr = cfg['lr'] * (1.0 - current_iter / max(total_iters, 1)) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr
            iter_num += 1
            global_step = iter_num

            total_loss.update(loss.item())
            total_loss_m1.update(model1_loss.item())
            total_loss_m2.update(model2_loss.item())
            total_consistency.update(consistency_weight)
            mask_ratio = valid_mask_u.float().mean().item()
            total_mask_ratio.update(mask_ratio)

            writer.add_scalar('train/loss_all', loss.item(), global_step)
            writer.add_scalar('train/model1_loss', model1_loss.item(), global_step)
            writer.add_scalar('train/model2_loss', model2_loss.item(), global_step)
            writer.add_scalar('train/consistency_weight', consistency_weight, global_step)
            writer.add_scalar('train/mask_ratio', mask_ratio, global_step)
            writer.add_scalar('train/lr', lr, global_step)

            if i % log_interval == 0:
                logger.info(
                    'Iters: {:}/{:}, Total loss: {:.3f}, model1 loss: {:.3f}, model2 loss: {:.3f}, Consistency: {:.3f}, Mask: {:.3f}'.format(
                        global_step,
                        total_iters,
                        total_loss.avg,
                        total_loss_m1.avg,
                        total_loss_m2.avg,
                        total_consistency.avg,
                        total_mask_ratio.avg,
                    )
                )

        is_best = False
        if iter_num >= total_iters * args.val_start and epoch % args.val_interval == 0:
            mDice1 = evaluate_model(model1, valloader, cfg, class_names, 'ABD_model1', writer, epoch, logger)
            mDice2 = evaluate_model(model2, valloader, cfg, class_names, 'ABD_model2', writer, epoch, logger)
            if mDice1.item() >= pre_best_dice1:
                pre_best_dice1 = mDice1.item()
                best_epoch1 = epoch
                torch.save({'model': model1.state_dict()}, os.path.join(cp_path, f'best_model1_ep{epoch}_m{pre_best_dice1:.3f}.pth'))
                is_best = True
            if mDice2.item() >= pre_best_dice2:
                pre_best_dice2 = mDice2.item()
                best_epoch2 = epoch
                torch.save({'model': model2.state_dict()}, os.path.join(cp_path, f'best_model2_ep{epoch}_m{pre_best_dice2:.3f}.pth'))
                is_best = True

        checkpoint = {
            'model1': model1.state_dict(),
            'model2': model2.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'epoch': epoch,
            'iter_num': iter_num,
            'start_time': start_time,
            'previous_best1': pre_best_dice1,
            'previous_best2': pre_best_dice2,
            'best_epoch1': best_epoch1,
            'best_epoch2': best_epoch2,
        }
        torch.save(checkpoint, latest_path)
        if is_best:
            logger.info(
                '*** best checkpoint updated: model1 {:.3f} @ {}, model2 {:.3f} @ {}'.format(
                    pre_best_dice1, best_epoch1, pre_best_dice2, best_epoch2
                )
            )

    end_time = time.time()
    logger.info('Training time: {:.2f}s'.format((end_time - start_time)))
    gc.collect()
    torch.cuda.empty_cache()
    writer.close()


if __name__ == '__main__':
    data_parser = argparse.ArgumentParser(description='datasets')
    data_parser.add_argument('--cli_dataset', type=str)
    known_args, remaining_args = data_parser.parse_known_args()
    dataset_name = known_args.cli_dataset
    if dataset_name not in DATASET_CONFIGS:
        print(f"Error: {dataset_name} not found in configs.")
        sys.exit(1)
    parser = get_parser(dataset_name)
    args = parser.parse_args(remaining_args)

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cp_path = os.path.join(
        args.checkpoint_path,
        'ABD/Ep{}bs{}_{}_seed{}_label{}/cons{}_{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, args.consistency, args.exp
        ),
    )
    os.makedirs(cp_path, exist_ok=True)
    save_path = os.path.join(cp_path, 'log')
    os.makedirs(save_path, exist_ok=True)

    include_list = ['comparison', 'utils', 'configs', 'Datasets', 'models', 'scripts', 'tools']
    target_dir = cp_path + '/code'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    for item in include_list:
        s = os.path.join('.', item)
        d = os.path.join(target_dir, item)
        if os.path.exists(s):
            if os.path.isdir(s):
                shutil.copytree(s, d, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                shutil.copy2(s, d)

    main(args, cfg, save_path, cp_path)
