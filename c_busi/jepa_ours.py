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
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.efficient import BUSISemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import count_params, init_log, AverageMeter, DiceLoss
from utils.val import eval_2d


DEFAULT_BUSI_CFG = {
    'dataset': 'BUSI',
    'base_dir': '/data/lhy_data/BUSI',
    'labelnum': 100,
    'label_help': 'number of labeled BUSI images',
    'num': None,
    'nclass': 2,
    'lr': 0.001,
    'epochs': 200,
    'batch_size': 8,
    'crop_size': 256,
    'conf_thresh': 0.95,
    'sup_weight': 1.0,
    'ot_weight': 0.5,
    'mask_ratio': 0.5,
    'mask_block_size': 32,
    'feature_dim': 64,
    'transformer_dim': 128,
    'transformer_depth': 4,
    'transformer_heads': 4,
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
        defaults.setdefault('dataset', cli_dataset)
        return defaults
    return dict(DEFAULT_BUSI_CFG)


def get_parser(defaults):
    parser = argparse.ArgumentParser(description=defaults.get('dataset', 'BUSI'))
    parser.add_argument('--dataset', type=str, default=defaults.get('dataset', 'BUSI'))
    parser.add_argument('--base_dir', type=str, default=defaults.get('base_dir', DEFAULT_BUSI_CFG['base_dir']))
    parser.add_argument('--labelnum', type=int, default=defaults.get('labelnum', DEFAULT_BUSI_CFG['labelnum']), help=defaults.get('label_help', DEFAULT_BUSI_CFG['label_help']))
    parser.add_argument('--num', default=defaults.get('num', DEFAULT_BUSI_CFG['num']), type=int, help='unlabeled data number')
    parser.add_argument('--config', type=str, default=defaults.get('config'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=str, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    return parser


def load_cfg(args, defaults):
    cfg = dict(DEFAULT_BUSI_CFG)
    cfg.update(defaults)
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg.update(yaml.load(f, Loader=yaml.Loader))
    cfg['dataset'] = args.dataset
    return cfg


def build_class_names(cfg):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(str(cfg.get('dataset')).upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def generate_2d_mask(feature, mask_ratio=0.5, min_block_size=16, max_block_size=None):
    b, _, h, w = feature.shape
    device = feature.device
    context_mask = torch.ones((b, 1, h, w), device=device, dtype=feature.dtype)
    target_mask = torch.zeros_like(context_mask)
    total_pixels = h * w
    target_pixels = max(1, int(total_pixels * mask_ratio))
    max_block_size = max_block_size or max(min_block_size, min(h, w) // 2)
    max_block_size = max(1, min(max_block_size, min(h, w)))
    min_block_size = max(1, min(min_block_size, max_block_size))

    for idx in range(b):
        masked_pixels = 0
        attempts = 0
        while masked_pixels < target_pixels and attempts < 64:
            block_h = random.randint(min_block_size, max_block_size)
            block_w = random.randint(min_block_size, max_block_size)
            top = random.randint(0, max(h - block_h, 0))
            left = random.randint(0, max(w - block_w, 0))
            region = context_mask[idx, :, top:top + block_h, left:left + block_w]
            newly_masked = region.sum().item()
            region.zero_()
            masked_pixels += int(newly_masked)
            attempts += 1

        if masked_pixels == 0:
            center_h = max(1, h // 4)
            center_w = max(1, w // 4)
            top = max(0, (h - center_h) // 2)
            left = max(0, (w - center_w) // 2)
            context_mask[idx, :, top:top + center_h, left:left + center_w] = 0

    target_mask = 1.0 - context_mask
    return context_mask, target_mask


class UNetFeatureBranch(nn.Module):
    def __init__(self, in_chns, class_num, feature_dim):
        super().__init__()
        self.backbone = UNet(in_chns=in_chns, class_num=class_num)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(16, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

    def forward(self, x, return_feature=False):
        logits, feature = self.backbone(x, use_feature=True)
        feature = self.feature_proj(feature)
        if return_feature:
            return logits, feature
        return logits


class TransformerSegBranch(nn.Module):
    def __init__(self, in_chns, class_num, feature_dim, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chns, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.pos_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, feature_dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(feature_dim, class_num, kernel_size=1)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

    def forward(self, x, return_feature=False):
        feature = self.patch_embed(x)
        feature = feature + self.pos_conv(feature)
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        feature = tokens.transpose(1, 2).reshape(b, c, h, w)
        feature = self.decoder(feature)
        logits = self.classifier(feature)
        latent_feature = self.feature_proj(feature)
        if return_feature:
            return logits, latent_feature
        return logits


class DecoupledPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or max(in_channels, 64)
        out_channels = out_channels or in_channels
        self.semantic_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        self.structural_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        semantic = self.semantic_head(x).expand(-1, -1, x.shape[-2], x.shape[-1])
        high_freq = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        structural = self.structural_head(high_freq)
        return self.fuse(torch.cat([semantic, structural], dim=1)) + self.residual(x)


class SinkhornOTLoss(nn.Module):
    def __init__(self, epsilon=0.05, max_iter=30, max_tokens=256):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_tokens = max_tokens

    def _sample_tokens(self, tokens):
        if tokens.shape[0] <= self.max_tokens:
            return tokens
        indices = torch.linspace(0, tokens.shape[0] - 1, steps=self.max_tokens, device=tokens.device).long()
        return tokens.index_select(0, indices)

    def _flatten_tokens(self, feature_map, mask=None):
        tokens = feature_map.permute(1, 2, 0).reshape(-1, feature_map.shape[0])
        if mask is not None:
            valid = mask.reshape(-1) > 0.5
            if valid.sum() > 1:
                tokens = tokens[valid]
        return self._sample_tokens(tokens)

    def _sinkhorn_distance(self, pred_tokens, target_tokens):
        pred_tokens = F.normalize(pred_tokens, dim=-1)
        target_tokens = F.normalize(target_tokens, dim=-1)
        cost = 1.0 - pred_tokens @ target_tokens.transpose(0, 1)
        log_kernel = -cost / self.epsilon
        log_mu = pred_tokens.new_full((pred_tokens.shape[0],), -np.log(pred_tokens.shape[0]))
        log_nu = target_tokens.new_full((target_tokens.shape[0],), -np.log(target_tokens.shape[0]))
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)

        for _ in range(self.max_iter):
            u = log_mu - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(log_kernel + u.unsqueeze(1), dim=0)

        transport = torch.exp(log_kernel + u.unsqueeze(1) + v.unsqueeze(0))
        return torch.sum(transport * cost)

    def forward(self, pred_feature, target_feature, target_mask=None):
        batch_losses = []
        for batch_idx in range(pred_feature.shape[0]):
            mask = None
            if target_mask is not None:
                mask = target_mask[batch_idx, 0] if target_mask.ndim == 4 else target_mask[batch_idx]
            pred_tokens = self._flatten_tokens(pred_feature[batch_idx], mask)
            target_tokens = self._flatten_tokens(target_feature[batch_idx], mask)
            if pred_tokens.shape[0] == 0 or target_tokens.shape[0] == 0:
                continue
            batch_losses.append(self._sinkhorn_distance(pred_tokens, target_tokens))

        if not batch_losses:
            return pred_feature.new_tensor(0.0)
        return torch.stack(batch_losses).mean()


def compute_supervised_loss(logits, target, criterion_ce, criterion_dice):
    loss_ce = criterion_ce(logits, target)
    loss_dice = criterion_dice(logits, target)
    return (loss_ce + loss_dice) / 2.0


def compute_bidirectional_ot_loss(model_context, model_target, predictor, context_input, target_input, target_mask, ot_loss):
    _, context_feature = model_context(context_input, return_feature=True)
    _, target_feature = model_target(target_input, return_feature=True)
    target_mask = F.interpolate(target_mask.float(), size=target_feature.shape[-2:], mode='nearest')
    predicted_target = predictor(context_feature)
    return ot_loss(predicted_target, target_feature.detach(), target_mask=target_mask)


def train_step(
    labeled_data,
    unlabeled_data,
    model_a,
    model_b,
    predictor_a2b,
    predictor_b2a,
    optimizer,
    criterion_ce,
    criterion_dice,
    ot_loss,
    cfg,
):
    img_x, mask_x = labeled_data
    img_x = img_x.cuda(non_blocking=True)
    mask_x = mask_x.cuda(non_blocking=True)
    img_w, img_s, ignore_mask, _ = unlabeled_data
    img_w = img_w.cuda(non_blocking=True)
    img_s = img_s.cuda(non_blocking=True)
    ignore_mask = ignore_mask.cuda(non_blocking=True)

    logits_a = model_a(img_x)
    logits_b = model_b(img_x)
    loss_sup_a = compute_supervised_loss(logits_a, mask_x, criterion_ce, criterion_dice)
    loss_sup_b = compute_supervised_loss(logits_b, mask_x, criterion_ce, criterion_dice)
    loss_sup = 0.5 * (loss_sup_a + loss_sup_b)

    context_mask, target_mask = generate_2d_mask(
        img_w,
        mask_ratio=cfg['mask_ratio'],
        min_block_size=cfg['mask_block_size'],
    )
    valid_mask = (ignore_mask != 255).unsqueeze(1).float()
    context_mask = context_mask * valid_mask
    target_mask = target_mask * valid_mask
    fallback_mask = (target_mask.sum(dim=(1, 2, 3), keepdim=True) == 0).float()
    target_mask = target_mask + fallback_mask * valid_mask
    target_mask = (target_mask > 0).float()

    loss_ot_a2b = compute_bidirectional_ot_loss(
        model_context=model_a,
        model_target=model_b,
        predictor=predictor_a2b,
        context_input=img_w * context_mask,
        target_input=img_s * target_mask,
        target_mask=target_mask,
        ot_loss=ot_loss,
    )
    loss_ot_b2a = compute_bidirectional_ot_loss(
        model_context=model_b,
        model_target=model_a,
        predictor=predictor_b2a,
        context_input=img_s * context_mask,
        target_input=img_w * target_mask,
        target_mask=target_mask,
        ot_loss=ot_loss,
    )
    loss_ot = 0.5 * (loss_ot_a2b + loss_ot_b2a)
    loss = cfg['sup_weight'] * loss_sup + cfg['ot_weight'] * loss_ot

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        unlabeled_fg_a = (model_a(img_w).argmax(dim=1) > 0).float().mean()
        unlabeled_fg_b = (model_b(img_s).argmax(dim=1) > 0).float().mean()

    return {
        'loss': loss.detach(),
        'loss_sup': loss_sup.detach(),
        'loss_sup_a': loss_sup_a.detach(),
        'loss_sup_b': loss_sup_b.detach(),
        'loss_ot': loss_ot.detach(),
        'loss_ot_a2b': loss_ot_a2b.detach(),
        'loss_ot_b2a': loss_ot_b2a.detach(),
        'mask_ratio': target_mask.mean().detach(),
        'pseudo_fg_a': unlabeled_fg_a.detach(),
        'pseudo_fg_b': unlabeled_fg_b.detach(),
    }


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model_a = init_2d_weight(UNetFeatureBranch(in_chns=3, class_num=cfg['nclass'], feature_dim=cfg['feature_dim']).cuda()).cuda()
    model_b = init_2d_weight(
        TransformerSegBranch(
            in_chns=3,
            class_num=cfg['nclass'],
            feature_dim=cfg['feature_dim'],
            embed_dim=cfg['transformer_dim'],
            depth=cfg['transformer_depth'],
            num_heads=cfg['transformer_heads'],
        ).cuda()
    ).cuda()
    predictor_a2b = init_2d_weight(DecoupledPredictor(cfg['feature_dim']).cuda()).cuda()
    predictor_b2a = init_2d_weight(DecoupledPredictor(cfg['feature_dim']).cuda()).cuda()

    optimizer = AdamW(
        params=list(model_a.parameters()) + list(model_b.parameters()) + list(predictor_a2b.parameters()) + list(predictor_b2a.parameters()),
        lr=cfg['lr'],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()
    ot_loss = SinkhornOTLoss(
        epsilon=cfg['sinkhorn_eps'],
        max_iter=cfg['sinkhorn_iters'],
        max_tokens=cfg['sinkhorn_max_tokens'],
    ).cuda()

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('UNet branch params: {:.3f}M'.format(count_params(model_a)))
    logger.info('Transformer branch params: {:.3f}M'.format(count_params(model_b)))
    logger.info('Predictor params: {:.3f}M'.format(count_params(predictor_a2b) + count_params(predictor_b2a)))

    trainset_u = BUSISemiDataset('train_u', args, cfg['crop_size'])
    trainset_l = BUSISemiDataset('train_l', args, cfg['crop_size'])
    valset = BUSISemiDataset('val', args, cfg['crop_size'])
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

    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d', total_iters)
    class_names = build_class_names(cfg)
    log_interval = max(len(trainloader) // 8, 1)
    best_dice_a = 0.0
    best_dice_b = 0.0
    best_epoch_a = 0
    best_epoch_b = 0
    epoch = -1
    iter_num = 0
    latest_path = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model_a.load_state_dict(checkpoint['model_a'])
        model_b.load_state_dict(checkpoint['model_b'])
        predictor_a2b.load_state_dict(checkpoint['predictor_a2b'])
        predictor_b2a.load_state_dict(checkpoint['predictor_b2a'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_dice_a = checkpoint['best_dice_a']
        best_dice_b = checkpoint['best_dice_b']
        best_epoch_a = checkpoint['best_epoch_a']
        best_epoch_b = checkpoint['best_epoch_b']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        start_time = time.time()

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum: {args.labelnum}, Previous best mdice '
            f'UNet: {best_dice_a:.4f} @epoch: {best_epoch_a}, Transformer: {best_dice_b:.4f} @epoch: {best_epoch_b}'
        )
        total_loss = AverageMeter()
        total_loss_sup = AverageMeter()
        total_loss_ot = AverageMeter()
        total_loss_sup_a = AverageMeter()
        total_loss_sup_b = AverageMeter()
        total_loss_ot_a2b = AverageMeter()
        total_loss_ot_b2a = AverageMeter()
        model_a.train()
        model_b.train()
        predictor_a2b.train()
        predictor_b2a.train()
        is_best = False

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            step_stats = train_step(
                labeled_data=labeled_data,
                unlabeled_data=unlabeled_data,
                model_a=model_a,
                model_b=model_b,
                predictor_a2b=predictor_a2b,
                predictor_b2a=predictor_b2a,
                optimizer=optimizer,
                criterion_ce=criterion_l,
                criterion_dice=diceloss,
                ot_loss=ot_loss,
                cfg=cfg,
            )

            total_loss.update(step_stats['loss'].item())
            total_loss_sup.update(step_stats['loss_sup'].item())
            total_loss_ot.update(step_stats['loss_ot'].item())
            total_loss_sup_a.update(step_stats['loss_sup_a'].item())
            total_loss_sup_b.update(step_stats['loss_sup_b'].item())
            total_loss_ot_a2b.update(step_stats['loss_ot_a2b'].item())
            total_loss_ot_b2a.update(step_stats['loss_ot_b2a'].item())
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            iter_num += 1

            writer.add_scalar('train/loss_all', step_stats['loss'].item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_sup', step_stats['loss_sup'].item(), iters)
            writer.add_scalar('train/loss_ot', step_stats['loss_ot'].item(), iters)
            writer.add_scalar('train/loss_sup_unet', step_stats['loss_sup_a'].item(), iters)
            writer.add_scalar('train/loss_sup_transformer', step_stats['loss_sup_b'].item(), iters)
            writer.add_scalar('train/loss_ot_a2b', step_stats['loss_ot_a2b'].item(), iters)
            writer.add_scalar('train/loss_ot_b2a', step_stats['loss_ot_b2a'].item(), iters)
            writer.add_scalar('train/masked_ratio', step_stats['mask_ratio'].item(), iters)
            writer.add_scalar('train/pseudo_fg_unet', step_stats['pseudo_fg_a'].item(), iters)
            writer.add_scalar('train/pseudo_fg_transformer', step_stats['pseudo_fg_b'].item(), iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {total_loss.avg:.3f}'
                    f', loss_sup: {total_loss_sup.avg:.3f}, loss_ot: {total_loss_ot.avg:.3f}'
                    f', sup_unet: {total_loss_sup_a.avg:.3f}, sup_trans: {total_loss_sup_b.avg:.3f}'
                    f', ot_a2b: {total_loss_ot_a2b.avg:.3f}, ot_b2a: {total_loss_ot_b2a.avg:.3f}'
                    f', masked_ratio: {step_stats["mask_ratio"].item():.4f}'
                )

        if iter_num >= total_iters * 0.3 and epoch % 2 == 0:
            model_a.eval()
            mDice_a, dice_class_a = eval_2d(valloader, model_a, cfg, ifdist=False, val_mode='unet')
            model_a.train()
            model_b.eval()
            mDice_b, dice_class_b = eval_2d(valloader, model_b, cfg, ifdist=False, val_mode='transformer')
            model_b.train()

            for cls_idx, dice in enumerate(dice_class_a):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info('*** Evaluation: Class [{:} {:}] Dice UNet: {:.3f}, Transformer: {:.3f}'.format(
                    cls_idx + 1, class_name, dice, dice_class_b[cls_idx]
                ))
                writer.add_scalar(f'eval/{class_name}_unet_DICE', dice, epoch)
                writer.add_scalar(f'eval/{class_name}_transformer_DICE', dice_class_b[cls_idx], epoch)

            logger.info('*** Evaluation: MeanDice UNet: {:.3f}, Transformer: {:.3f}'.format(mDice_a, mDice_b))
            writer.add_scalar('eval/mDice_unet', mDice_a.item(), epoch)
            writer.add_scalar('eval/mDice_transformer', mDice_b.item(), epoch)

            is_best = (mDice_a.item() >= best_dice_a) or (mDice_b.item() >= best_dice_b)
            best_dice_a = max(mDice_a.item(), best_dice_a)
            best_dice_b = max(mDice_b.item(), best_dice_b)
            if mDice_a.item() == best_dice_a:
                best_epoch_a = epoch
            if mDice_b.item() == best_dice_b:
                best_epoch_b = epoch

        checkpoint = {
            'model_a': model_a.state_dict(),
            'model_b': model_b.state_dict(),
            'predictor_a2b': predictor_a2b.state_dict(),
            'predictor_b2a': predictor_b2a.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_dice_a': best_dice_a,
            'best_dice_b': best_dice_b,
            'best_epoch_a': best_epoch_a,
            'best_epoch_b': best_epoch_b,
            'iter_num': iter_num,
            'start_time': start_time,
        }
        torch.save(checkpoint, latest_path)
        model_ckpt = {
            'model_a': model_a.state_dict(),
            'model_b': model_b.state_dict(),
            'predictor_a2b': predictor_a2b.state_dict(),
            'predictor_b2a': predictor_b2a.state_dict(),
        }
        if is_best:
            logger.info(
                '*** best checkpoint: MeanDice UNet: {:.3f}, Transformer: {:.3f}\n*** exp: {}'.format(
                    best_dice_a, best_dice_b, args.exp
                )
            )
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_unet_{best_dice_a:.3f}_trans_{best_dice_b:.3f}.pth'))

        if epoch >= cfg['epochs'] - 1:
            end_time = time.time()
            logger.info('Training time: {:.2f}s'.format((end_time - start_time)))
            gc.collect()
            torch.cuda.empty_cache()
            writer.close()


if __name__ == '__main__':
    data_parser = argparse.ArgumentParser(description='datasets')
    data_parser.add_argument('--cli_dataset', type=str, default='BUSI')
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
