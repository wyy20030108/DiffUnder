import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')

import argparse
import gc
import logging
import pprint
import random
import shutil
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from Datasets.efficient import ACDCsemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import AverageMeter, DiceLoss, count_params, init_log
from utils.val import eval_2d


DEFAULT_ACDC_CFG = {
    'dataset': 'ACDC',
    'base_dir': '/data/lhy_data/ACDC',
    'labelnum': 14,
    'label_help': '14 for 1:5; 7 for 1:10; 3 for 1:20',
    'num': 56,
    'nclass': 4,
    'lr': 0.01,
    'epochs': 200,
    'batch_size': 8,
    'crop_size': [256, 256],
    'val_patch_size': [256, 256],
    'num_workers': 6,
    'conf_thresh': 0.95,
    'sup_weight': 1.0,
    'consistency_weight': 0.5,
    'consistency_mode': 'l2',
    'mask_ratio': 0.2,
    'max_mask_ratio': 0.45,
    'mask_block_size': 32,
    'entropy_thresh': 0.15,
    'ignore_conf_thresh': 0.7,
    'feature_drop_prob': 0.7,
    'feature_noise_std': 0.15,
    'feature_dim': 64,
    'sinkhorn_eps': 0.05,
    'sinkhorn_iters': 30,
    'sinkhorn_max_tokens': 256,
}


def init_2d_weight(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(buffer.data)


def get_default_args(cli_dataset):
    if cli_dataset in DATASET_CONFIGS:
        defaults = dict(DATASET_CONFIGS[cli_dataset])
        defaults.setdefault('dataset', 'ACDC')
        return defaults
    return dict(DEFAULT_ACDC_CFG)


def get_parser(defaults):
    parser = argparse.ArgumentParser(description=defaults.get('dataset', 'ACDC'))
    parser.add_argument('--dataset', type=str, default=defaults.get('dataset', 'ACDC'))
    parser.add_argument('--base_dir', type=str, default=defaults.get('base_dir', DEFAULT_ACDC_CFG['base_dir']))
    parser.add_argument(
        '--labelnum',
        type=int,
        default=defaults.get('labelnum', DEFAULT_ACDC_CFG['labelnum']),
        help=defaults.get('label_help', DEFAULT_ACDC_CFG['label_help']),
    )
    parser.add_argument('--num', default=defaults.get('num', DEFAULT_ACDC_CFG['num']), type=int, help='unlabeled data number')
    parser.add_argument('--config', type=str, default=defaults.get('config'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=str, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    return parser


def load_cfg(args, defaults):
    cfg = dict(DEFAULT_ACDC_CFG)
    cfg.update(defaults)
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg.update(yaml.load(f, Loader=yaml.Loader))
    cfg['dataset'] = cfg.get('dataset', 'ACDC')
    cfg.setdefault('consistency_weight', cfg.get('ot_weight', DEFAULT_ACDC_CFG['consistency_weight']))
    cfg.setdefault('consistency_mode', DEFAULT_ACDC_CFG['consistency_mode'])
    return cfg


def build_class_names(cfg):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(str(cfg.get('dataset')).upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def compute_teacher_confidence(teacher_logits):
    probs = torch.softmax(teacher_logits, dim=1)
    confidence = probs.amax(dim=1, keepdim=True)
    entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(dim=1, keepdim=True)
    entropy = entropy / np.log(teacher_logits.shape[1])
    return probs, confidence, entropy


def generate_confidence_guided_mask(confidence_score, candidate_mask, mask_ratios, min_block_size=16, max_block_size=None):
    b, _, h, w = confidence_score.shape
    target_mask = torch.zeros_like(candidate_mask, dtype=confidence_score.dtype)
    max_block_size = max_block_size or max(min_block_size, min(h, w) // 4)
    max_block_size = max(1, min(max_block_size, min(h, w)))
    min_block_size = max(1, min(min_block_size, max_block_size))

    for idx in range(b):
        candidate = candidate_mask[idx, 0] > 0.5
        candidate_count = int(candidate.sum().item())
        if candidate_count == 0:
            continue

        target_pixels = max(1, int(candidate_count * float(mask_ratios[idx].item())))
        masked_pixels = 0
        attempts = 0

        while masked_pixels < target_pixels and attempts < 96:
            best_region = None
            best_score = None
            for _ in range(6):
                block_h = random.randint(min_block_size, max_block_size)
                block_w = random.randint(min_block_size, max_block_size)
                top = random.randint(0, max(h - block_h, 0))
                left = random.randint(0, max(w - block_w, 0))
                region_candidate = candidate[top:top + block_h, left:left + block_w]
                if not region_candidate.any():
                    continue
                region_score = confidence_score[idx, 0, top:top + block_h, left:left + block_w][region_candidate].mean().item()
                if best_region is None or region_score > best_score:
                    best_region = (top, left, block_h, block_w)
                    best_score = region_score

            if best_region is None:
                break

            top, left, block_h, block_w = best_region
            region_target = target_mask[idx, 0, top:top + block_h, left:left + block_w]
            region_candidate = candidate[top:top + block_h, left:left + block_w]
            region_new = region_candidate & (region_target < 0.5)
            newly_masked = int(region_new.sum().item())
            if newly_masked == 0:
                attempts += 1
                continue
            region_target[region_new] = 1.0
            masked_pixels += newly_masked
            attempts += 1

        if masked_pixels < target_pixels:
            remaining_mask = candidate & (target_mask[idx, 0] < 0.5)
            remaining_count = int(remaining_mask.sum().item())
            if remaining_count > 0:
                remaining = min(target_pixels - masked_pixels, remaining_count)
                coords = remaining_mask.nonzero(as_tuple=False)
                scores = confidence_score[idx, 0, coords[:, 0], coords[:, 1]]
                selected = torch.topk(scores, k=remaining).indices
                chosen = coords[selected]
                target_mask[idx, 0, chosen[:, 0], chosen[:, 1]] = 1.0

    return target_mask


def build_confidence_guided_targets(teacher_logits, valid_mask, cfg):
    _, confidence, entropy = compute_teacher_confidence(teacher_logits)
    high_conf_mask = (
        (confidence >= cfg['conf_thresh'])
        & (entropy <= cfg['entropy_thresh'])
        & (valid_mask > 0)
    ).float()

    valid_pixels = valid_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mean_confidence = (confidence * valid_mask).sum(dim=(1, 2, 3)) / valid_pixels
    mean_entropy = (entropy * valid_mask).sum(dim=(1, 2, 3)) / valid_pixels
    reliable_pixels = high_conf_mask.sum(dim=(1, 2, 3))
    reliable_ratio = reliable_pixels / valid_pixels

    conf_progress = ((mean_confidence - cfg['ignore_conf_thresh']) / (1.0 - cfg['ignore_conf_thresh'])).clamp(0.0, 1.0)
    entropy_progress = (1.0 - mean_entropy / max(cfg['entropy_thresh'], 1e-6)).clamp(0.0, 1.0)
    dynamic_ratio = cfg['mask_ratio'] + (cfg['max_mask_ratio'] - cfg['mask_ratio']) * (0.5 * conf_progress + 0.5 * entropy_progress)
    dynamic_ratio = dynamic_ratio * (reliable_ratio > 0).float()
    dynamic_ratio = dynamic_ratio.clamp(min=0.0, max=cfg['max_mask_ratio'])

    confidence_score = confidence * (1.0 - entropy) * high_conf_mask
    target_mask = generate_confidence_guided_mask(
        confidence_score=confidence_score,
        candidate_mask=high_conf_mask,
        mask_ratios=dynamic_ratio,
        min_block_size=cfg['mask_block_size'],
    )
    return {
        'target_mask': target_mask,
        'high_conf_mask': high_conf_mask,
        'confidence': confidence,
        'entropy': entropy,
        'mean_confidence': mean_confidence.mean(),
        'mean_entropy': mean_entropy.mean(),
        'reliable_ratio': reliable_ratio.mean(),
        'dynamic_mask_ratio': dynamic_ratio.mean(),
    }


def apply_feature_perturbation(feature, perturb_mask, drop_prob=0.7, noise_std=0.15):
    if perturb_mask is None:
        return feature

    spatial_mask = F.interpolate(perturb_mask.float(), size=feature.shape[-2:], mode='nearest')
    if spatial_mask.max().item() <= 0:
        return feature

    spatial_mask = spatial_mask.expand_as(feature)
    feature_mean = feature.mean(dim=(2, 3), keepdim=True)
    feature_std = feature.detach().flatten(2).std(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    noise = torch.randn_like(feature) * feature_std * noise_std
    noisy_feature = feature + noise

    if drop_prob > 0:
        drop_mask = (torch.rand_like(perturb_mask.float()) < drop_prob).float()
        drop_mask = F.interpolate(drop_mask, size=feature.shape[-2:], mode='nearest').expand_as(feature) * spatial_mask
    else:
        drop_mask = spatial_mask

    perturbed_feature = feature * (1.0 - spatial_mask) + noisy_feature * spatial_mask
    fill_feature = feature_mean + noise
    perturbed_feature = perturbed_feature * (1.0 - drop_mask) + fill_feature * drop_mask
    return perturbed_feature


class UNetFeatureBranch(nn.Module):
    def __init__(self, in_chns, class_num, feature_dim):
        super().__init__()
        self.backbone = UNet(in_chns=in_chns, class_num=class_num)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(16, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x,
        return_feature=False,
        perturb_mask=None,
        feature_drop_prob=0.0,
        feature_noise_std=0.0,
    ):
        logits, feature = self.backbone(x, use_feature=True)
        if perturb_mask is not None and (feature_drop_prob > 0 or feature_noise_std > 0):
            feature = apply_feature_perturbation(
                feature,
                perturb_mask=perturb_mask,
                drop_prob=feature_drop_prob,
                noise_std=feature_noise_std,
            )
            logits = self.backbone.classifier(feature)
        feature = self.feature_proj(feature)
        if return_feature:
            return logits, feature
        return logits


def compute_supervised_loss(logits, target, criterion_ce, criterion_dice):
    loss_ce = criterion_ce(logits, target)
    loss_dice = criterion_dice(logits, target)
    return (loss_ce + loss_dice) / 2.0


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


def compute_pixel_consistency_loss(student_map, teacher_map, target_mask=None, mode='l2'):
    teacher_map = teacher_map.detach()
    if mode == 'kl':
        loss_map = F.kl_div(
            F.log_softmax(student_map, dim=1),
            F.softmax(teacher_map, dim=1),
            reduction='none',
        ).sum(dim=1, keepdim=True)
    else:
        loss_map = (student_map - teacher_map).pow(2).mean(dim=1, keepdim=True)

    if target_mask is not None:
        target_mask = F.interpolate(target_mask.float(), size=loss_map.shape[-2:], mode='nearest')
        denom = target_mask.sum().clamp_min(1.0)
        return (loss_map * target_mask).sum() / denom
    return loss_map.mean()


def train_step(
    labeled_data,
    unlabeled_data,
    student_model,
    teacher_model,
    optimizer,
    criterion_ce,
    criterion_dice,
    cfg,
    ema_decay,
    global_step,
):
    img_x, mask_x = labeled_data
    img_x = img_x.cuda(non_blocking=True)
    mask_x = mask_x.cuda(non_blocking=True)
    img_w, img_s, ignore_mask, _ = unlabeled_data
    img_w = img_w.cuda(non_blocking=True)
    img_s = img_s.cuda(non_blocking=True)
    ignore_mask = ignore_mask.cuda(non_blocking=True)

    student_logits = student_model(img_x)
    loss_sup = compute_supervised_loss(student_logits, mask_x, criterion_ce, criterion_dice)

    valid_mask = (ignore_mask != 255).unsqueeze(1).float()
    with torch.no_grad():
        teacher_model.eval()
        teacher_unlabeled_logits, teacher_unlabeled_feature = teacher_model(img_w, return_feature=True)
        confidence_targets = build_confidence_guided_targets(teacher_unlabeled_logits, valid_mask, cfg)

    target_mask = confidence_targets['target_mask']
    student_unlabeled_logits, student_unlabeled_feature = student_model(
        img_s,
        return_feature=True,
        perturb_mask=target_mask,
        feature_drop_prob=cfg['feature_drop_prob'],
        feature_noise_std=cfg['feature_noise_std'],
    )

    loss_feature = compute_pixel_consistency_loss(
        student_unlabeled_feature,
        teacher_unlabeled_feature,
        target_mask=target_mask,
        mode=cfg['consistency_mode'],
    )
    loss_logits = compute_pixel_consistency_loss(
        student_unlabeled_logits,
        teacher_unlabeled_logits,
        target_mask=target_mask,
        mode='kl',
    )
    loss_consistency = loss_feature + 0.25 * loss_logits
    loss = cfg['sup_weight'] * loss_sup + cfg['consistency_weight'] * loss_consistency

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    update_ema_variables(student_model, teacher_model, ema_decay, global_step)

    with torch.no_grad():
        teacher_fg = (teacher_unlabeled_logits.argmax(dim=1) > 0).float().mean()
        student_fg = (student_unlabeled_logits.argmax(dim=1) > 0).float().mean()

    return {
        'loss': loss.detach(),
        'loss_sup': loss_sup.detach(),
        'loss_consistency': loss_consistency.detach(),
        'loss_feature': loss_feature.detach(),
        'loss_logits': loss_logits.detach(),
        'mask_ratio': target_mask.mean().detach(),
        'dynamic_mask_ratio': confidence_targets['dynamic_mask_ratio'].detach(),
        'reliable_ratio': confidence_targets['reliable_ratio'].detach(),
        'mean_confidence': confidence_targets['mean_confidence'].detach(),
        'mean_entropy': confidence_targets['mean_entropy'].detach(),
        'pseudo_fg_student': student_fg.detach(),
        'pseudo_fg_teacher': teacher_fg.detach(),
    }


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    student_model = init_2d_weight(
        UNetFeatureBranch(in_chns=1, class_num=cfg['nclass'], feature_dim=cfg['feature_dim']).cuda()
    ).cuda()
    teacher_model = deepcopy(student_model).cuda()
    set_requires_grad(teacher_model, False)
    teacher_model.eval()

    optimizer = AdamW(
        params=student_model.parameters(),
        lr=cfg['lr'],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Student params: {:.3f}M'.format(count_params(student_model)))
    logger.info('Teacher params: {:.3f}M'.format(count_params(teacher_model)))

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
        num_workers=cfg.get('num_workers', DEFAULT_ACDC_CFG['num_workers']),
        pin_memory=True,
        collate_fn=mix_collate_fn,
    )

    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d', total_iters)
    class_names = build_class_names(cfg)
    log_interval = max(len(trainloader) // 8, 1)
    best_dice_student = 0.0
    best_dice_teacher = 0.0
    best_epoch_student = 0
    best_epoch_teacher = 0
    epoch = -1
    iter_num = 0
    latest_path = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        try:
            if 'student_model' in checkpoint:
                student_model.load_state_dict(checkpoint['student_model'])
                teacher_model.load_state_dict(checkpoint['teacher_model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                epoch = checkpoint['epoch']
                best_dice_student = checkpoint['best_dice_student']
                best_dice_teacher = checkpoint['best_dice_teacher']
                best_epoch_student = checkpoint['best_epoch_student']
                best_epoch_teacher = checkpoint['best_epoch_teacher']
                iter_num = checkpoint['iter_num']
                start_time = checkpoint['start_time']
                logger.info('************ Load EMA checkpoint at epoch %i\n' % epoch)
            else:
                student_model.load_state_dict(checkpoint['model'], strict=False)
                teacher_model.load_state_dict(checkpoint.get('model_ema', checkpoint['model']), strict=False)
                start_time = time.time()
                logger.info('************ Load legacy student weights and reset optimizer state\n')
        except Exception as exc:
            start_time = time.time()
            logger.warning('Checkpoint load failed, restart from scratch: %s', exc)
    else:
        start_time = time.time()

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum: {args.labelnum}, Previous best mdice '
            f'Student: {best_dice_student:.4f} @epoch: {best_epoch_student}, Teacher: {best_dice_teacher:.4f} @epoch: {best_epoch_teacher}'
        )
        total_loss = AverageMeter()
        total_loss_sup = AverageMeter()
        total_loss_consistency = AverageMeter()
        total_loss_feature = AverageMeter()
        total_loss_logits = AverageMeter()
        student_model.train()
        teacher_model.eval()
        is_best = False

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            step_stats = train_step(
                labeled_data=labeled_data,
                unlabeled_data=unlabeled_data,
                student_model=student_model,
                teacher_model=teacher_model,
                optimizer=optimizer,
                criterion_ce=criterion_l,
                criterion_dice=diceloss,
                cfg=cfg,
                ema_decay=args.ema_decay,
                global_step=iter_num + 1,
            )

            total_loss.update(step_stats['loss'].item())
            total_loss_sup.update(step_stats['loss_sup'].item())
            total_loss_consistency.update(step_stats['loss_consistency'].item())
            total_loss_feature.update(step_stats['loss_feature'].item())
            total_loss_logits.update(step_stats['loss_logits'].item())
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            iter_num += 1

            writer.add_scalar('train/loss_all', step_stats['loss'].item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_sup', step_stats['loss_sup'].item(), iters)
            writer.add_scalar('train/loss_consistency', step_stats['loss_consistency'].item(), iters)
            writer.add_scalar('train/loss_feature', step_stats['loss_feature'].item(), iters)
            writer.add_scalar('train/loss_logits', step_stats['loss_logits'].item(), iters)
            writer.add_scalar('train/masked_ratio', step_stats['mask_ratio'].item(), iters)
            writer.add_scalar('train/dynamic_mask_ratio', step_stats['dynamic_mask_ratio'].item(), iters)
            writer.add_scalar('train/reliable_ratio', step_stats['reliable_ratio'].item(), iters)
            writer.add_scalar('train/mean_confidence', step_stats['mean_confidence'].item(), iters)
            writer.add_scalar('train/mean_entropy', step_stats['mean_entropy'].item(), iters)
            writer.add_scalar('train/pseudo_fg_student', step_stats['pseudo_fg_student'].item(), iters)
            writer.add_scalar('train/pseudo_fg_teacher', step_stats['pseudo_fg_teacher'].item(), iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, loss: {total_loss.avg:.3f}'
                    f', l_l: {total_loss_sup.avg:.3f}, l_consis: {total_loss_consistency.avg:.3f}'
                    f', feature_l2: {total_loss_feature.avg:.3f}, logits_kl: {total_loss_logits.avg:.3f}'
                    f', masked_ratio: {step_stats["mask_ratio"].item():.4f}'
                    f', dyn_ratio: {step_stats["dynamic_mask_ratio"].item():.4f}'
                    f', relia_ratio: {step_stats["reliable_ratio"].item():.4f}'
                    f', mean_conf: {step_stats["mean_confidence"].item():.4f}'
                    f', mean_ent: {step_stats["mean_entropy"].item():.4f}'
                )

        if iter_num >= total_iters * 0.3 and epoch % 2 == 0:
            student_model.eval()
            mDice_student, dice_class_student = eval_2d(valloader, student_model, cfg, ifdist=False, val_mode='student')
            student_model.train()
            teacher_model.eval()
            mDice_teacher, dice_class_teacher = eval_2d(valloader, teacher_model, cfg, ifdist=False, val_mode='teacher')

            for cls_idx, dice in enumerate(dice_class_student):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info(
                    '*** Evaluation: Class [{:} {:}] Dice Student: {:.3f}, Teacher: {:.3f}'.format(
                        cls_idx + 1, class_name, dice, dice_class_teacher[cls_idx]
                    )
                )
                writer.add_scalar(f'eval/{class_name}_student_DICE', dice, epoch)
                writer.add_scalar(f'eval/{class_name}_teacher_DICE', dice_class_teacher[cls_idx], epoch)

            logger.info('*** Evaluation: MeanDice Student: {:.3f}, Teacher: {:.3f}'.format(mDice_student, mDice_teacher))
            writer.add_scalar('eval/mDice_student', mDice_student.item(), epoch)
            writer.add_scalar('eval/mDice_teacher', mDice_teacher.item(), epoch)

            is_best = (mDice_student.item() >= best_dice_student) or (mDice_teacher.item() >= best_dice_teacher)
            best_dice_student = max(mDice_student.item(), best_dice_student)
            best_dice_teacher = max(mDice_teacher.item(), best_dice_teacher)
            if mDice_student.item() == best_dice_student:
                best_epoch_student = epoch
            if mDice_teacher.item() == best_dice_teacher:
                best_epoch_teacher = epoch

        checkpoint = {
            'student_model': student_model.state_dict(),
            'teacher_model': teacher_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_dice_student': best_dice_student,
            'best_dice_teacher': best_dice_teacher,
            'best_epoch_student': best_epoch_student,
            'best_epoch_teacher': best_epoch_teacher,
            'iter_num': iter_num,
            'start_time': start_time,
        }
        torch.save(checkpoint, latest_path)
        model_ckpt = {
            'student_model': student_model.state_dict(),
            'teacher_model': teacher_model.state_dict(),
        }
        if is_best:
            logger.info(
                '*** best checkpoint: MeanDice Student: {:.3f}, Teacher: {:.3f}\n*** exp: {}'.format(
                    best_dice_student, best_dice_teacher, args.exp
                )
            )
            torch.save(
                model_ckpt,
                os.path.join(cp_path, f'ep{epoch}_student_{best_dice_student:.3f}_teacher_{best_dice_teacher:.3f}.pth'),
            )

        if epoch >= cfg['epochs'] - 1:
            end_time = time.time()
            logger.info('Training time: {:.2f}s'.format((end_time - start_time)))
            gc.collect()
            torch.cuda.empty_cache()
            writer.close()


if __name__ == '__main__':
    data_parser = argparse.ArgumentParser(description='datasets')
    data_parser.add_argument('--cli_dataset', type=str, default='20acdc')
    known_args, remaining_args = data_parser.parse_known_args()
    defaults = get_default_args(known_args.cli_dataset)
    parser = get_parser(defaults)
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
    cfg = load_cfg(args, defaults)

    cp_path = os.path.join(
        args.checkpoint_path,
        'OT_JEPA/Ep{}bs{}_{}_seed{}_label{}/{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, args.exp
        ),
    )
    os.makedirs(cp_path, exist_ok=True)
    save_path = os.path.join(cp_path, 'log')
    os.makedirs(save_path, exist_ok=True)

    include_list = ['Datasets', 'models', 'utils', 'configs', 'c_busi', 'tools']
    target_dir = cp_path + '/code'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    for item in include_list:
        src = os.path.join('.', item)
        dst = os.path.join(target_dir, item)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                shutil.copy2(src, dst)

    main(args, cfg, save_path, cp_path)
