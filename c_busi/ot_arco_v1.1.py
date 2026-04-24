# 计算当前状态（有标签数据）：对于有标签数据集 $\mathcal{D}_L$ 中的当前 Batch，对于任意给定的真实类别（Ground Truth） $i \in \{0, 1, ..., C-1\}$，我们提取所有属于类别 $i$ 的像素的集合 $\Omega_i$。计算网络对这些像素输出的平均概率分布向量 $v_i \in \mathbb{R}^C$：$$v_{i, j} = \frac{1}{|\Omega_i|} \sum_{x \in \Omega_i} P(y=j | x)$$其中 $P(y=j | x)$ 是模型（通常是 Teacher/EMA 模型）输出的 Softmax 概率。EMA 记录趋势：在每个 Step，利用 EMA 将 $v_i$ 融入矩阵 $M$ 的第 $i$ 行：$$M_{i}^{(t)} = \alpha \cdot M_{i}^{(t-1)} + (1 - \alpha) \cdot v_i$$注：此时 $M_i^{(t)}$ 完美记录了你所说的趋势。例如，如果 $i$ 代表类别 A，在初期 $M_i$ 可能是 $[0.8, 0.1, 0.1]$（趋近背景）；到了中期，因为边界模糊，它变成了 $[0.05, 0.75, 0.20]$（逐渐趋近A，但保留 20% 类别 B 的语义）。对齐无标签数据（知识迁移）：对于无标签像素 $u$，假设我们通过最佳阈值 $\tau$ 或 Sinkhorn 算法赋予了它一个伪标签 $\hat{y}_u = c$。我们不再使用 One-hot 向量作为它的学习目标，而是使用矩阵 $M$ 的第 $c$ 行作为其软标签（Soft Target）：$$P_{target}(u) = M_{c}^{(t)}$$最后，使用 KL 散度让 Student 模型输出 $P_S(u)$ 对齐这个带有历史趋势的软标签
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')

import argparse
import gc
import logging
import math
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
from torch.optim import SGD, AdamW
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.efficient import BUSISemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import AverageMeter, DiceLoss, count_params, init_log
from utils.val import eval_2d


DEFAULT_BUSI_CFG = {
    'dataset': 'BUSI',
    'base_dir': '/data/lhy_data/BUSI/Dataset_BUSI_with_GT/without_normal',
    'labelnum': 64,
    'label_help': '64 for 1:8; 32 for 1:16; 16 for 1:32',
    'num': None,
    'nclass': 2,
    'lr': 0.01,
    'epochs': 200,
    'batch_size': 8,
    'crop_size': [256, 256],
    'val_patch_size': [256, 256],
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


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
    parser.add_argument('--exp', type=str, required=True, help='Experiment description')
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.99)

    parser.add_argument('--strong_threshold', type=float, default=0.97)
    parser.add_argument('--strong_threshold_u2pl', type=float, default=0.97)
    parser.add_argument('--weak_threshold', type=float, default=0.7)
    parser.add_argument('--apply_aug', type=str, default='cutmix', choices=['cutout', 'cutmix', 'classmix', 'none'])
    parser.add_argument('--K', type=int, default=36, help='revisiting queue size')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--num_negatives', type=int, default=512)
    parser.add_argument('--num_queries', type=int, default=256)
    parser.add_argument('--reco_weight', type=float, default=0.01)
    parser.add_argument('--unsup_weight', type=float, default=1.0)
    parser.add_argument('--revisit_weight', type=float, default=1.0)
    parser.add_argument('--reco_temperature', type=float, default=0.5)
    parser.add_argument('--func', type=str, default='smc', choices=['asmc', 'smc', 'rand'])
    parser.add_argument('--memobank_size', type=int, default=30000)
    parser.add_argument('--memobank_bg_size', type=int, default=50000)
    parser.add_argument('--ot_weight', type=float, default=0.5)
    parser.add_argument('--sinkhorn_eps', type=float, default=0.5)
    parser.add_argument('--sinkhorn_iters', type=int, default=5)
    parser.add_argument('--sinkhorn_relax', type=float, default=10.0)
    parser.add_argument('--sinkhorn_relax_min', type=float, default=0.1)
    parser.add_argument('--sinkhorn_fg_prob_threshold', type=float, default=0.05)
    parser.add_argument('--class_prior', type=str, default=None)
    parser.add_argument('--use_fixed_prior', type=str2bool, default=None)
    parser.add_argument('--prior_momentum', type=float, default=0.99)
    parser.add_argument('--prototype_momentum', type=float, default=0.99)
    parser.add_argument('--prototype_threshold', type=float, default=0.8)
    parser.add_argument('--prototype_min_threshold', type=float, default=0.5)
    parser.add_argument('--prototype_temperature', type=float, default=0.5)
    parser.add_argument('--num_prototypes_per_class', type=int, default=4)
    parser.add_argument('--rampup_epochs', type=float, default=40.0)
    parser.add_argument('--sharpen_temperature', type=float, default=0.5)
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


class FeatureExtractor(nn.Module):
    def __init__(self, fea_dim, output_dim):
        super().__init__()
        assert len(fea_dim) == 5, 'FeatureExtractor expects 5 feature scales.'
        cnt = fea_dim[0]
        self.fea0 = nn.Conv2d(cnt, cnt, kernel_size=1, bias=False)
        cnt += fea_dim[1]
        self.fea1 = nn.Conv2d(cnt, cnt, kernel_size=1, bias=False)
        cnt += fea_dim[2]
        self.fea2 = nn.Conv2d(cnt, cnt, kernel_size=1, bias=False)
        cnt += fea_dim[3]
        self.fea3 = nn.Conv2d(cnt, cnt, kernel_size=1, bias=False)
        cnt += fea_dim[4]
        self.fea4 = nn.Conv2d(cnt, output_dim, kernel_size=1, bias=False)

    def forward(self, fea_list):
        x = self.fea0(fea_list[0]) + fea_list[0]
        x = F.interpolate(x, size=fea_list[1].shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, fea_list[1]), dim=1)
        x = self.fea1(x) + x
        x = F.interpolate(x, size=fea_list[2].shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, fea_list[2]), dim=1)
        x = self.fea2(x) + x
        x = F.interpolate(x, size=fea_list[3].shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, fea_list[3]), dim=1)
        x = self.fea3(x) + x
        x = F.interpolate(x, size=fea_list[4].shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, fea_list[4]), dim=1)
        return self.fea4(x)


class RepresentationHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(output_dim, output_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.proj(x)


def forward_with_features(model, x):
    features = model.encoder(x)
    decoder_feature = model.decoder(features)
    logits = model.classifier(decoder_feature)
    return logits, list(reversed(features))


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(buffer.data)


@torch.no_grad()
def momentum_update_module(student_module, teacher_module, momentum):
    for teacher_param, student_param in zip(teacher_module.parameters(), student_module.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)
    for teacher_buffer, student_buffer in zip(teacher_module.buffers(), student_module.buffers()):
        teacher_buffer.data.copy_(student_buffer.data)


@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr):
    if keys.numel() == 0:
        return
    if keys.shape[0] >= queue.shape[0]:
        queue.copy_(keys[-queue.shape[0]:])
        queue_ptr[0] = 0
        return
    ptr = int(queue_ptr.item())
    batch_size = keys.shape[0]
    end = ptr + batch_size
    if end <= queue.shape[0]:
        queue[ptr:end] = keys
    else:
        first = queue.shape[0] - ptr
        queue[ptr:] = keys[:first]
        queue[:end - queue.shape[0]] = keys[first:]
    queue_ptr[0] = end % queue.shape[0]


@torch.no_grad()
def enqueue_memobank(keys, queue, queue_ptr, queue_size):
    if keys.numel() == 0:
        return 0
    keys = keys.detach().cpu()
    queue[0] = torch.cat((queue[0], keys), dim=0)
    if queue[0].shape[0] > queue_size:
        queue[0] = queue[0][-queue_size:]
    queue_ptr[0] = min(queue[0].shape[0], queue_size)
    return keys.shape[0]


def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio
    min_w = max(1, int(img_size[1] / ratio) + 1)
    w = int(torch.randint(min_w, img_size[1] + 1, (1,)).item())
    h = int(round(cutout_area / max(w, 1)))
    h = max(1, min(h, img_size[0]))
    x_start = int(torch.randint(0, img_size[1] - w + 1, (1,)).item())
    y_start = int(torch.randint(0, img_size[0] - h + 1, (1,)).item())
    x_end = int(x_start + w)
    y_end = int(y_start + h)
    mask = torch.ones(img_size, dtype=torch.float32)
    mask[y_start:y_end, x_start:x_end] = 0.0
    return mask


def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)
    if labels.numel() <= 1:
        return torch.ones_like(pseudo_labels, dtype=torch.float32)
    labels_select = labels[torch.randperm(len(labels), device=labels.device)][:max(1, len(labels) // 2)]
    return (pseudo_labels.unsqueeze(-1) == labels_select).any(-1).float()


def generate_unsup_data(data, target, logits, mode='cutmix'):
    if mode == 'none':
        return data, target.long(), logits

    batch_size, _, im_h, im_w = data.shape
    device = data.device
    new_data, new_target, new_logits = [], [], []

    for i in range(batch_size):
        if mode == 'cutout':
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
            cutout_target = target[i].clone()
            cutout_target[(1 - mix_mask).bool()] = -1
            new_data.append((data[i] * mix_mask).unsqueeze(0))
            new_target.append(cutout_target.unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue

        if mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        elif mode == 'classmix':
            mix_mask = generate_class_mask(target[i]).to(device)
        else:
            mix_mask = torch.ones((im_h, im_w), device=device)

        pair_index = (i + 1) % batch_size
        mix_mask_bool = mix_mask.bool()
        mixed_target = torch.where(mix_mask_bool, target[i], target[pair_index])
        mixed_logits = torch.where(mix_mask_bool, logits[i], logits[pair_index])
        mixed_data = data[i] * mix_mask + data[pair_index] * (1 - mix_mask)
        new_data.append(mixed_data.unsqueeze(0))
        new_target.append(mixed_target.unsqueeze(0))
        new_logits.append(mixed_logits.unsqueeze(0))

    return torch.cat(new_data), torch.cat(new_target).long(), torch.cat(new_logits)


def _expand_spatial_mask(mask, tensor):
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(0)
    return mask


def _apply_unsup_aug_tensor(tensor, paired_tensor, mix_mask, mode, fill_value):
    mix_mask = _expand_spatial_mask(mix_mask.bool(), tensor)
    if mode == 'cutout':
        fill_tensor = torch.full_like(tensor, fill_value)
        return torch.where(mix_mask, tensor, fill_tensor)
    return torch.where(mix_mask, tensor, paired_tensor)


def generate_unsup_data_with_soft_targets(data, target, logits, soft_target, ignore_mask, mode='cutmix'):
    if mode == 'none':
        return data, target.long(), logits, soft_target, ignore_mask

    batch_size, _, im_h, im_w = data.shape
    device = data.device
    new_data, new_target, new_logits = [], [], []
    new_soft_target, new_ignore_mask = [], []

    for i in range(batch_size):
        if mode == 'cutout':
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
        elif mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        elif mode == 'classmix':
            mix_mask = generate_class_mask(target[i]).to(device)
        else:
            mix_mask = torch.ones((im_h, im_w), device=device)

        pair_index = (i + 1) % batch_size
        new_data.append(_apply_unsup_aug_tensor(data[i], data[pair_index], mix_mask, mode, 0.0).unsqueeze(0))
        new_target.append(_apply_unsup_aug_tensor(target[i], target[pair_index], mix_mask, mode, -1).unsqueeze(0))
        new_logits.append(_apply_unsup_aug_tensor(logits[i], logits[pair_index], mix_mask, mode, 0.0).unsqueeze(0))
        new_soft_target.append(_apply_unsup_aug_tensor(soft_target[i], soft_target[pair_index], mix_mask, mode, 0.0).unsqueeze(0))
        new_ignore_mask.append(_apply_unsup_aug_tensor(ignore_mask[i], ignore_mask[pair_index], mix_mask, mode, 255).unsqueeze(0))

    return (
        torch.cat(new_data),
        torch.cat(new_target).long(),
        torch.cat(new_logits),
        torch.cat(new_soft_target),
        torch.cat(new_ignore_mask).long(),
    )


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((batch_size, num_segments, im_h, im_w), device=inputs.device, dtype=torch.float32)
    inputs_temp = inputs.clone()
    invalid_mask = inputs_temp < 0
    inputs_temp[invalid_mask] = 0
    outputs.scatter_(1, inputs_temp.unsqueeze(1), 1.0)
    outputs[invalid_mask.unsqueeze(1).expand_as(outputs)] = 0
    return outputs


def percentile_threshold(values, percentile, default_value):
    if values.numel() == 0:
        return values.new_tensor(default_value)
    q = torch.tensor(percentile / 100.0, device=values.device, dtype=values.dtype)
    return torch.quantile(values.float(), q).to(values.dtype)


def normalize_distribution(distribution):
    distribution = distribution.float().clamp_min(1e-8)
    return distribution / distribution.sum().clamp_min(1e-8)


def sharpen_probability_map(prob_map, temperature=0.5):
    temperature = max(float(temperature), 1e-3)
    sharpened = prob_map.clamp_min(1e-8).pow(1.0 / temperature)
    return sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(1e-8)


def parse_class_prior(prior_source, num_classes):
    if prior_source is None:
        return None
    if isinstance(prior_source, str):
        cleaned = prior_source.strip().strip('[]')
        if not cleaned:
            return None
        values = [float(item.strip()) for item in cleaned.split(',') if item.strip()]
        prior = torch.tensor(values, dtype=torch.float32)
    else:
        prior = torch.as_tensor(prior_source, dtype=torch.float32)

    if prior.numel() == num_classes - 1:
        bg_prior = max(1e-6, 1.0 - prior.sum().item())
        prior = torch.cat((torch.tensor([bg_prior], dtype=torch.float32), prior), dim=0)
    if prior.numel() != num_classes:
        raise ValueError(f'class_prior length mismatch: expected {num_classes}, got {prior.numel()}')
    return normalize_distribution(prior)


def build_target_prior(args, cfg, num_classes, device):
    prior_source = args.class_prior if args.class_prior is not None else cfg.get('class_prior')
    use_fixed_prior = args.use_fixed_prior if args.use_fixed_prior is not None else cfg.get('use_fixed_prior')
    if prior_source is None:
        return torch.full((num_classes,), 1.0 / num_classes, device=device), bool(use_fixed_prior) if use_fixed_prior is not None else False
    prior = parse_class_prior(prior_source, num_classes).to(device)
    return prior, True if use_fixed_prior is None else bool(use_fixed_prior)


@torch.no_grad()
def update_running_prior(running_prior, labels, num_classes, momentum):
    valid_labels = labels[(labels >= 0) & (labels != 255)]
    if valid_labels.numel() == 0:
        return
    batch_hist = torch.bincount(valid_labels.view(-1), minlength=num_classes).float().to(running_prior.device)
    batch_hist = normalize_distribution(batch_hist)
    running_prior.mul_(momentum).add_(batch_hist, alpha=1 - momentum)
    running_prior.copy_(normalize_distribution(running_prior))


def gaussian_rampup(current, rampup_length):
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - current / rampup_length
    return math.exp(-5.0 * phase * phase)


@torch.no_grad()
def sinkhorn_align_distribution(prob_map, valid_mask, target_prior, eps=0.5, num_iters=5, relax=10.0):
    batch_size, num_classes, height, width = prob_map.shape
    flat_prob = prob_map.permute(0, 2, 3, 1).reshape(-1, num_classes)
    flat_valid = valid_mask.reshape(-1).bool()
    if flat_valid.sum() == 0:
        return prob_map.detach()

    prob_valid = flat_prob[flat_valid].detach()
    kernel = torch.exp(torch.log(prob_valid.clamp_min(1e-6)) / max(eps, 1e-3)).clamp_min(1e-8)
    row_prior = torch.full((kernel.shape[0],), 1.0 / max(kernel.shape[0], 1), device=kernel.device, dtype=kernel.dtype)
    col_prior = normalize_distribution(target_prior.to(kernel.device, kernel.dtype))
    relax = max(float(relax), 1e-3)
    relax_power = relax / (relax + max(float(eps), 1e-3))
    transport = kernel
    for _ in range(num_iters):
        row_scale = (row_prior / transport.sum(dim=1).clamp_min(1e-8)).pow(relax_power).unsqueeze(1)
        col_scale = (col_prior / transport.sum(dim=0).clamp_min(1e-8)).pow(relax_power).unsqueeze(0)
        transport = transport * row_scale
        transport = transport * col_scale
    transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-8)

    aligned = flat_prob.clone()
    aligned[flat_valid] = transport
    return aligned.view(batch_size, height, width, num_classes).permute(0, 3, 1, 2)


@torch.no_grad()
def compute_adaptive_sinkhorn_relax(prob_map, valid_mask, base_relax, min_relax=0.1, fg_prob_threshold=0.05):
    base_relax = max(float(base_relax), 1e-3)
    min_relax = min(base_relax, max(float(min_relax), 1e-3))
    if prob_map.shape[1] <= 1:
        return base_relax

    flat_valid = valid_mask.reshape(-1).bool()
    if flat_valid.sum() == 0:
        return min_relax

    fg_prob = prob_map[:, 1:].sum(dim=1).reshape(-1)[flat_valid].mean().item()
    threshold = max(float(fg_prob_threshold), 1e-6)
    relax_ratio = min(max(fg_prob / threshold, 0.0), 1.0)
    return min_relax + (base_relax - min_relax) * relax_ratio


def compute_distribution_alignment_loss(logits, soft_target, valid_mask):
    per_pixel = -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1)
    valid = valid_mask.float()
    return (per_pixel * valid).sum() / valid.sum().clamp_min(1.0)


@torch.no_grad()
def build_classwise_prototype_mask(confidence_map, label_map, valid_mask, relative_ratio=0.8, min_threshold=0.5):
    prototype_valid = torch.zeros_like(valid_mask, dtype=torch.bool)
    valid_mask = valid_mask.bool()
    if not valid_mask.any():
        return prototype_valid

    for cls_idx in torch.unique(label_map[valid_mask]).tolist():
        cls_mask = valid_mask & (label_map == cls_idx)
        cls_conf = confidence_map[cls_mask]
        if cls_conf.numel() == 0:
            continue
        cls_threshold = max(float(min_threshold), cls_conf.max().item() * float(relative_ratio))
        prototype_valid |= cls_mask & confidence_map.ge(cls_threshold)
    return prototype_valid


@torch.no_grad()
def update_ema_prototypes(
    prototypes,
    prototype_mask,
    features,
    labels,
    valid_mask,
    momentum,
    num_prototypes_per_class=4,
    assignment_temp=0.5,
):
    feature_map = F.normalize(features.permute(0, 2, 3, 1), dim=-1)
    valid_mask = valid_mask.bool()
    for cls_idx in range(prototypes.shape[0]):
        cls_mask = valid_mask & (labels == cls_idx)
        if not cls_mask.any():
            continue
        cls_features = feature_map[cls_mask]
        max_proto = min(prototypes.shape[1], 1 if cls_idx == 0 else max(1, num_prototypes_per_class))

        if not prototype_mask[cls_idx, :max_proto].any():
            init_idx = torch.randint(cls_features.shape[0], (max_proto,), device=cls_features.device)
            init_proto = F.normalize(cls_features[init_idx], dim=1)
            prototypes[cls_idx, :max_proto] = init_proto
            prototype_mask[cls_idx, :max_proto] = True
            continue

        active_idx = torch.nonzero(prototype_mask[cls_idx, :max_proto], as_tuple=False).flatten()
        cls_proto = F.normalize(prototypes[cls_idx, active_idx], dim=1)
        assign_logits = torch.matmul(cls_features, cls_proto.t()) / max(assignment_temp, 1e-3)
        assign_weights = torch.softmax(assign_logits, dim=1)
        proto_mass = assign_weights.sum(dim=0, keepdim=True).t().clamp_min(1e-6)
        batch_proto = torch.matmul(assign_weights.t(), cls_features) / proto_mass
        batch_proto = F.normalize(batch_proto, dim=1)
        updated = momentum * prototypes[cls_idx, active_idx] + (1 - momentum) * batch_proto
        prototypes[cls_idx, active_idx] = F.normalize(updated, dim=1)


def compute_prototype_contrastive_loss(
    rep_l,
    mask_l,
    rep_u,
    pseudo_u,
    valid_u_mask,
    prototypes,
    prototype_mask,
    func='smc',
    num_queries=256,
    temp=0.5,
):
    available_proto = torch.nonzero(prototype_mask, as_tuple=False)
    if available_proto.shape[0] <= 1:
        return rep_l.sum() * 0.0

    sample_fn = {'asmc': grid_as_monte_carlo_sample, 'smc': grid_monte_carlo_sample}.get(func, monte_carlo_sample)
    proto_bank = F.normalize(prototypes[available_proto[:, 0], available_proto[:, 1]], dim=1)
    bank_classes = available_proto[:, 0]
    available_classes = torch.unique(bank_classes)

    feature_l = F.normalize(rep_l.permute(0, 2, 3, 1), dim=-1)
    feature_u = F.normalize(rep_u.permute(0, 2, 3, 1), dim=-1)
    losses = []

    for cls_idx in available_classes.tolist():
        feature_list = []
        labeled_mask = mask_l == cls_idx
        if labeled_mask.any():
            feature_list.append(feature_l[labeled_mask])

        unlabeled_mask = valid_u_mask & (pseudo_u == cls_idx)
        if unlabeled_mask.any():
            feature_list.append(feature_u[unlabeled_mask])

        if not feature_list:
            continue

        cls_features = torch.cat(feature_list, dim=0)
        sample_count = min(num_queries, cls_features.shape[0])
        sample_idx = sample_fn(cls_features.shape[0], sample_count, device=cls_features.device)
        anchor_features = cls_features[sample_idx]
        logits = torch.matmul(anchor_features, proto_bank.t()) / temp
        positive_mask = bank_classes == cls_idx
        if positive_mask.sum() == 0:
            continue
        with torch.no_grad():
            soft_targets = torch.softmax(logits[:, positive_mask], dim=1)
            full_targets = torch.zeros_like(logits)
            full_targets[:, positive_mask] = soft_targets
        losses.append((-(full_targets * F.log_softmax(logits, dim=1)).sum(dim=1)).mean())

    if not losses:
        return rep_l.sum() * 0.0
    return torch.stack(losses).mean()


def get_revisiting_loss(random_pool, rep_u, rep_u_teacher, topk=5):
    rep_u = F.adaptive_avg_pool2d(rep_u, 1).flatten(1)
    rep_u = F.normalize(rep_u, dim=-1)
    rep_u_teacher = F.adaptive_avg_pool2d(rep_u_teacher, 1).flatten(1)
    rep_u_teacher = F.normalize(rep_u_teacher, dim=-1)
    dist_student = 2 - 2 * torch.einsum('bc,kc->bk', rep_u, random_pool)
    dist_teacher = 2 - 2 * torch.einsum('bc,kc->bk', rep_u_teacher, random_pool)
    topk = min(topk, random_pool.shape[0])
    if topk <= 0:
        return rep_u.sum() * 0.0, rep_u_teacher.detach()
    _, nn_index = dist_teacher.topk(topk, dim=1, largest=False)
    nn_dist_student = torch.gather(dist_student, 1, nn_index)
    return (nn_dist_student.sum(dim=1) / topk).mean(), rep_u_teacher.detach()


def monte_carlo_sample(high, shape, device=None):
    device = torch.device('cpu') if device is None else device
    if high <= 0:
        return torch.zeros((shape,), dtype=torch.long, device=device)
    return torch.randint(high, size=(shape,), dtype=torch.long, device=device)


def grid_monte_carlo_sample(high, shape, cut_count=4, device=None):
    device = torch.device('cpu') if device is None else device
    if high <= 0:
        return torch.zeros((shape,), dtype=torch.long, device=device)
    try:
        edge = round(math.sqrt(high))
        if edge <= 1:
            return monte_carlo_sample(high, shape, device=device)
        samples = []
        per_patch_sample = max(1, shape * edge * edge // max(high, 1) // (cut_count ** 2))
        img = torch.arange(edge * edge, device=device, dtype=torch.long).reshape(edge, edge)
        patch_edge = max(1, edge // cut_count)
        for i in range(cut_count):
            for j in range(cut_count):
                if i != cut_count - 1 and j != cut_count - 1:
                    picked = img[i * patch_edge:(i + 1) * patch_edge, j * patch_edge:(j + 1) * patch_edge]
                elif i == cut_count - 1 and j != cut_count - 1:
                    picked = img[i * patch_edge:, j * patch_edge:(j + 1) * patch_edge]
                elif i != cut_count - 1 and j == cut_count - 1:
                    picked = img[i * patch_edge:(i + 1) * patch_edge, j * patch_edge:]
                else:
                    picked = img[i * patch_edge:, j * patch_edge:]
                picked = picked.flatten()
                if picked.numel() == 0:
                    continue
                picked = picked[torch.randperm(picked.numel(), device=device)]
                indices = torch.randint(0, picked.numel(), size=(per_patch_sample,), device=device)
                samples.append(picked[indices])
        if not samples:
            return monte_carlo_sample(high, shape, device=device)
        sample_tensor = torch.cat(samples, dim=0)
        sample_tensor = sample_tensor[sample_tensor < high]
        if sample_tensor.numel() < shape:
            pad = torch.randint(high, (shape - sample_tensor.numel(),), dtype=torch.long, device=device)
            sample_tensor = torch.cat((sample_tensor, pad), dim=0)
        return sample_tensor[torch.randperm(sample_tensor.numel(), device=device)][:shape]
    except Exception:
        return monte_carlo_sample(high, shape, device=device)


def grid_as_monte_carlo_sample(high, shape, cut_count=4, device=None):
    device = torch.device('cpu') if device is None else device
    if high <= 0:
        return torch.zeros((shape,), dtype=torch.long, device=device)
    try:
        edge = round(math.sqrt(high))
        if edge <= 1:
            return monte_carlo_sample(high, shape, device=device)
        samples = []
        per_patch_sample = max(2, shape * edge * edge // max(high, 1) // (cut_count ** 2))
        img = torch.arange(edge * edge, device=device, dtype=torch.long).reshape(edge, edge)
        patch_edge = max(1, edge // cut_count)
        for i in range(cut_count):
            for j in range(cut_count):
                if i != cut_count - 1 and j != cut_count - 1:
                    picked = img[i * patch_edge:(i + 1) * patch_edge, j * patch_edge:(j + 1) * patch_edge]
                elif i == cut_count - 1 and j != cut_count - 1:
                    picked = img[i * patch_edge:, j * patch_edge:(j + 1) * patch_edge]
                elif i != cut_count - 1 and j == cut_count - 1:
                    picked = img[i * patch_edge:(i + 1) * patch_edge, j * patch_edge:]
                else:
                    picked = img[i * patch_edge:, j * patch_edge:]
                picked = picked.flatten()
                if picked.numel() == 0:
                    continue
                picked = picked[torch.randperm(picked.numel(), device=device)]
                center = int((2 * picked.float().mean()).item())
                half = max(1, per_patch_sample // 2)
                indices = torch.randint(0, picked.numel(), size=(half,), device=device)
                chosen = picked[indices]
                mirrored = center - chosen
                samples.extend([chosen, mirrored])
        if not samples:
            return monte_carlo_sample(high, shape, device=device)
        sample_tensor = torch.cat(samples, dim=0)
        sample_tensor = sample_tensor[(sample_tensor >= 0) & (sample_tensor < high)]
        if sample_tensor.numel() < shape:
            pad = torch.randint(high, (shape - sample_tensor.numel(),), dtype=torch.long, device=device)
            sample_tensor = torch.cat((sample_tensor, pad), dim=0)
        return sample_tensor[torch.randperm(sample_tensor.numel(), device=device)][:shape]
    except Exception:
        return monte_carlo_sample(high, shape, device=device)


def compute_unsupervised_loss(predict, target, logits, strong_threshold, ignore_mask=None):
    batch_size = predict.shape[0]
    target = target.long()
    valid_mask = (target >= 0).float()
    if ignore_mask is not None:
        valid_mask = valid_mask * (ignore_mask != 255).float()
    denom = valid_mask.view(batch_size, -1).sum(-1).clamp_min(1.0)
    weighting = logits.view(batch_size, -1).ge(strong_threshold).float().sum(-1) / denom
    loss = F.cross_entropy(predict, target.clamp_min(0), reduction='none')
    return (weighting[:, None, None] * loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model = UNet(in_chns=3, class_num=cfg['nclass']).cuda()
    ema_model = deepcopy(model).cuda()
    for param in ema_model.parameters():
        param.detach_()

    feature_dims = list(reversed(model.encoder.ft_chns))
    feature_dim_sum = sum(feature_dims)
    q_feature_extractor = FeatureExtractor(feature_dims, feature_dim_sum).cuda()
    k_feature_extractor = deepcopy(q_feature_extractor).cuda()
    for param in k_feature_extractor.parameters():
        param.detach_()
    q_representation = RepresentationHead(feature_dim_sum, feature_dim_sum).cuda()

    params = list(model.parameters()) + list(q_feature_extractor.parameters()) + list(q_representation.parameters())
    optimizer = SGD(params=params, lr=cfg['lr'], weight_decay=1e-4, momentum=0.9, nesterov=True)

    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()

    num_gpus = torch.cuda.device_count()
    logger.info('use {} gpus!'.format(num_gpus))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

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
    log_interval = max(len(trainloader) // 8, 1)
    class_names = build_class_names(cfg)

    random_pool = F.normalize(torch.randn(args.K, feature_dim_sum, device='cuda'), dim=1)
    random_pool_ptr = torch.zeros(1, dtype=torch.long, device='cuda')
    class_prototypes = torch.zeros(
        (cfg['nclass'], max(1, args.num_prototypes_per_class), feature_dim_sum),
        dtype=torch.float32,
        device='cuda',
    )
    prototype_mask = torch.zeros(
        (cfg['nclass'], max(1, args.num_prototypes_per_class)),
        dtype=torch.bool,
        device='cuda',
    )
    target_prior, use_fixed_prior = build_target_prior(args, cfg, cfg['nclass'], class_prototypes.device)

    pre_best_dice = 0.0
    pre_best_dice_ema = 0.0
    best_epoch = 0
    best_epoch_ema = 0
    epoch = -1
    iter_num = 0

    latest_ckpt_path = os.path.join(cp_path, 'latest.pth')
    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        q_feature_extractor.load_state_dict(checkpoint['q_feature_extractor'])
        k_feature_extractor.load_state_dict(checkpoint['k_feature_extractor'])
        q_representation.load_state_dict(checkpoint['q_representation'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'random_pool' in checkpoint:
            random_pool.copy_(checkpoint['random_pool'].cuda())
        if 'random_pool_ptr' in checkpoint:
            random_pool_ptr.copy_(checkpoint['random_pool_ptr'].cuda())
        if 'class_prototypes' in checkpoint:
            ckpt_prototypes = checkpoint['class_prototypes'].cuda()
            if ckpt_prototypes.dim() == 2:
                class_prototypes[:, 0].copy_(ckpt_prototypes)
            else:
                copy_proto = min(class_prototypes.shape[1], ckpt_prototypes.shape[1])
                class_prototypes[:, :copy_proto].copy_(ckpt_prototypes[:, :copy_proto])
        if 'prototype_mask' in checkpoint:
            ckpt_mask = checkpoint['prototype_mask'].cuda()
            if ckpt_mask.dim() == 1:
                prototype_mask[:, 0].copy_(ckpt_mask)
            else:
                copy_proto = min(prototype_mask.shape[1], ckpt_mask.shape[1])
                prototype_mask[:, :copy_proto].copy_(ckpt_mask[:, :copy_proto])
        if 'target_prior' in checkpoint:
            target_prior.copy_(checkpoint['target_prior'].cuda())
        epoch = checkpoint['epoch']
        pre_best_dice = checkpoint['previous_best']
        pre_best_dice_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        logger.info('************ Load from checkpoint at epoch %i\n', epoch)
    else:
        start_time = time.time()

    for epoch in range(epoch + 1, cfg['epochs']):
        exp_name = args.exp.split('_')[-1] if args.exp else 'default'
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, {exp_name}, seed:{args.seed}, labelnum: {args.labelnum}, '
            f'Previous best mdice model: {pre_best_dice:.4f} @epoch: {best_epoch}, '
            f'ema: {pre_best_dice_ema:.4f} @epoch_ema: {best_epoch_ema}'
        )

        loss_meter = AverageMeter()
        sup_meter = AverageMeter()
        unsup_meter = AverageMeter()
        ot_meter = AverageMeter()
        proto_meter = AverageMeter()
        revisit_meter = AverageMeter()
        is_best = False
        mDice = torch.tensor(pre_best_dice)
        mDice_ema = torch.tensor(pre_best_dice_ema)

        model.train()
        ema_model.train()
        q_feature_extractor.train()
        k_feature_extractor.train()
        q_representation.train()

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            current_epoch = epoch + float(i) / max(len(trainloader), 1)
            ssl_rampup = gaussian_rampup(current_epoch, args.rampup_epochs)
            current_ot_weight = 0.0
            current_reco_weight = args.reco_weight * ssl_rampup

            img_x, mask_x = labeled_data
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)

            img_w, img_s, ignore_mask, _ = unlabeled_data
            img_w = img_w.cuda(non_blocking=True)
            img_s = img_s.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)

            with torch.no_grad():
                pred_u_teacher_weak, u_feature_map_teacher_weak = forward_with_features(ema_model, img_w)
                if not use_fixed_prior:
                    update_running_prior(target_prior, mask_x, cfg['nclass'], args.prior_momentum)

                weak_valid_u_mask = ignore_mask != 255
                pseudo_prob_weak = torch.softmax(pred_u_teacher_weak, dim=1)
                adaptive_relax = compute_adaptive_sinkhorn_relax(
                    prob_map=pseudo_prob_weak.detach(),
                    valid_mask=weak_valid_u_mask,
                    base_relax=args.sinkhorn_relax,
                    min_relax=args.sinkhorn_relax_min,
                    fg_prob_threshold=args.sinkhorn_fg_prob_threshold,
                )
                aligned_prob_u_weak = sinkhorn_align_distribution(
                    pseudo_prob_weak.detach(),
                    weak_valid_u_mask,
                    target_prior,
                    eps=args.sinkhorn_eps,
                    num_iters=args.sinkhorn_iters,
                    relax=adaptive_relax,
                )
                aligned_conf_u_weak, aligned_label_u_weak = torch.max(aligned_prob_u_weak, dim=1)
                train_u_aug_data, train_u_aug_label, train_u_aug_logits, train_u_aug_soft_label, train_u_aug_ignore_mask = generate_unsup_data_with_soft_targets(
                    img_s,
                    aligned_label_u_weak.clone(),
                    aligned_conf_u_weak.clone(),
                    aligned_prob_u_weak.clone(),
                    ignore_mask.clone(),
                    mode=args.apply_aug,
                )

            pred_l, l_feature_map = forward_with_features(model, img_x)
            pred_u, u_feature_map = forward_with_features(model, train_u_aug_data)

            with torch.no_grad():
                pred_l_teacher, l_feature_map_teacher = forward_with_features(ema_model, img_x)

            l_feature_all = q_feature_extractor(l_feature_map)
            u_feature_all = q_feature_extractor(u_feature_map)
            l_feature_all_teacher = k_feature_extractor(l_feature_map_teacher)
            u_feature_all_teacher = k_feature_extractor(u_feature_map_teacher_weak)

            rep_l = q_representation(l_feature_all)
            rep_u = q_representation(u_feature_all)
            rep_l_teacher = l_feature_all_teacher
            rep_u_teacher = u_feature_all_teacher

            loss_ce = criterion_l(pred_l, mask_x)
            loss_dice = diceloss(pred_l, mask_x)
            supervised_loss = loss_ce + loss_dice

            train_u_aug_soft_label = sharpen_probability_map(
                train_u_aug_soft_label.detach(),
                temperature=args.sharpen_temperature,
            )
            valid_u_mask = (
                (train_u_aug_label >= 0)
                & (train_u_aug_ignore_mask != 255)
                & train_u_aug_logits.ge(args.strong_threshold)
            )
            unsup_loss = compute_distribution_alignment_loss(
                pred_u,
                train_u_aug_soft_label,
                valid_u_mask,
            )

            with torch.no_grad():
                prototype_valid_u_weak = build_classwise_prototype_mask(
                    aligned_conf_u_weak,
                    aligned_label_u_weak,
                    weak_valid_u_mask,
                    relative_ratio=args.prototype_threshold,
                    min_threshold=args.prototype_min_threshold,
                )
                prototype_valid_u = build_classwise_prototype_mask(
                    train_u_aug_logits,
                    train_u_aug_label,
                    valid_u_mask,
                    relative_ratio=args.prototype_threshold,
                    min_threshold=args.prototype_min_threshold,
                )
                labeled_valid = torch.ones_like(mask_x, dtype=torch.bool)
                update_ema_prototypes(
                    class_prototypes,
                    prototype_mask,
                    torch.cat((rep_l_teacher, rep_u_teacher), dim=0).detach(),
                    torch.cat((mask_x, aligned_label_u_weak), dim=0),
                    torch.cat((labeled_valid, prototype_valid_u_weak), dim=0),
                    args.prototype_momentum,
                    num_prototypes_per_class=args.num_prototypes_per_class,
                    assignment_temp=args.prototype_temperature,
                )

            ot_loss = pred_u.new_tensor(0.0)
            proto_loss = compute_prototype_contrastive_loss(
                rep_l=rep_l,
                mask_l=mask_x,
                rep_u=rep_u,
                pseudo_u=train_u_aug_label,
                valid_u_mask=prototype_valid_u,
                prototypes=class_prototypes.detach(),
                prototype_mask=prototype_mask.detach(),
                func=args.func,
                num_queries=args.num_queries,
                temp=args.prototype_temperature,
            )

            loss_q, pooled_rep_u_teacher = get_revisiting_loss(random_pool, rep_u, rep_u_teacher, topk=args.topk)

            loss = (
                supervised_loss
                + args.unsup_weight * unsup_loss
                + current_reco_weight * proto_loss
                + args.revisit_weight * loss_q
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dequeue_and_enqueue(pooled_rep_u_teacher, random_pool, random_pool_ptr)

            iter_num += 1
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            momentum_update_module(q_feature_extractor, k_feature_extractor, args.ema_decay)

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss_meter.update(loss.item())
            sup_meter.update(supervised_loss.item())
            unsup_meter.update(unsup_loss.item())
            ot_meter.update(ot_loss.item())
            proto_meter.update(proto_loss.item())
            revisit_meter.update(loss_q.item())

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/loss_sup', supervised_loss.item(), iters)
            writer.add_scalar('train/loss_unsup', unsup_loss.item(), iters)
            writer.add_scalar('train/loss_ot', ot_loss.item(), iters)
            writer.add_scalar('train/loss_proto', proto_loss.item(), iters)
            writer.add_scalar('train/loss_revisit', loss_q.item(), iters)
            writer.add_scalar('train/ssl_rampup', ssl_rampup, iters)
            writer.add_scalar('train/weight_ot', current_ot_weight, iters)
            writer.add_scalar('train/weight_reco', current_reco_weight, iters)
            writer.add_scalar('train/pseudo_fg_ratio', (train_u_aug_label > 0).float().mean().item(), iters)
            writer.add_scalar('train/pseudo_conf_ratio', train_u_aug_logits.ge(args.strong_threshold).float().mean().item(), iters)
            writer.add_scalar('train/pseudo_conf_valid_ratio', valid_u_mask.float().mean().item(), iters)
            writer.add_scalar('train/prior_fg_ratio', target_prior[1:].sum().item(), iters)
            writer.add_scalar('train/sinkhorn_relax', adaptive_relax, iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {loss_meter.avg:.3f}, '
                    f'sup: {sup_meter.avg:.3f}, unsup: {unsup_meter.avg:.3f}, '
                    f'ot: {ot_meter.avg:.3f}, proto: {proto_meter.avg:.3f}, '
                    f'revisit: {revisit_meter.avg:.3f}, l_q: {loss_q.item():.3f}, '
                    f'ot_w: {current_ot_weight:.3f}, reco_w: {current_reco_weight:.4f}'
                )

        if iter_num >= total_iters * 0.3 and epoch % 2 == 0:
            model.eval()
            mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode='model')
            model.train()
            ema_model.eval()
            mDice_ema, dice_class_ema = eval_2d(valloader, ema_model, cfg, ifdist=False, val_mode='ema')
            ema_model.train()

            mDice = torch.as_tensor(mDice)
            mDice_ema = torch.as_tensor(mDice_ema)
            for cls_idx, dice in enumerate(dice_class):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info(
                    '*** Evaluation: Class [{:} {:}] Dice model: {:.3f}, ema: {:.3f}'.format(
                        cls_idx + 1,
                        class_name,
                        dice,
                        dice_class_ema[cls_idx],
                    )
                )
                writer.add_scalar(f'eval/{class_name}_model_DICE', float(dice), epoch)
                writer.add_scalar(f'eval/{class_name}_ema_DICE', float(dice_class_ema[cls_idx]), epoch)

            logger.info('*** Evaluation: MeanDice model: %.3f, ema: %.3f', mDice.item(), mDice_ema.item())
            writer.add_scalar('eval/mDice_model', mDice.item(), epoch)
            writer.add_scalar('eval/mDice_ema', mDice_ema.item(), epoch)

            is_best = (mDice.item() >= pre_best_dice) or (mDice_ema.item() >= pre_best_dice_ema)
            pre_best_dice = max(mDice.item(), pre_best_dice)
            pre_best_dice_ema = max(mDice_ema.item(), pre_best_dice_ema)
            if mDice.item() == pre_best_dice:
                best_epoch = epoch
            if mDice_ema.item() == pre_best_dice_ema:
                best_epoch_ema = epoch

        checkpoint = {
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'q_feature_extractor': q_feature_extractor.state_dict(),
            'k_feature_extractor': k_feature_extractor.state_dict(),
            'q_representation': q_representation.state_dict(),
            'optimizer': optimizer.state_dict(),
            'random_pool': random_pool.detach().cpu(),
            'random_pool_ptr': random_pool_ptr.detach().cpu(),
            'class_prototypes': class_prototypes.detach().cpu(),
            'prototype_mask': prototype_mask.detach().cpu(),
            'target_prior': target_prior.detach().cpu(),
            'epoch': epoch,
            'previous_best': pre_best_dice,
            'previous_best_ema': pre_best_dice_ema,
            'best_epoch': best_epoch,
            'best_epoch_ema': best_epoch_ema,
            'iter_num': iter_num,
            'start_time': start_time,
        }
        torch.save(checkpoint, latest_ckpt_path)
        model_ckpt = {
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'q_feature_extractor': q_feature_extractor.state_dict(),
            'k_feature_extractor': k_feature_extractor.state_dict(),
            'q_representation': q_representation.state_dict(),
        }
        if is_best:
            logger.info(
                '*** best checkpoint: MeanDice model: {:.3f}, ema: {:.3f}\n*** exp: {}'.format(
                    mDice.item(),
                    mDice_ema.item(),
                    exp_name,
                )
            )
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_m{mDice.item():.3f}_ema{mDice_ema.item():.3f}.pth'))

        if epoch >= cfg['epochs'] - 1:
            end_time = time.time()
            logger.info('Training time: %.2fs', end_time - start_time)
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
        'OT_ARCO_BUSI/Ep{}bs{}_{}_seed{}_label{}/{}'.format(
            cfg['epochs'],
            cfg['batch_size'],
            cfg['dataset'],
            args.seed,
            args.labelnum,
            args.exp,
        ),
    )
    os.makedirs(cp_path, exist_ok=True)
    save_path = os.path.join(cp_path, 'log')
    os.makedirs(save_path, exist_ok=True)

    include_list = ['Datasets', 'models', 'utils', 'configs', 'c_busi', 'tools']
    target_dir = os.path.join(cp_path, 'code')
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
        else:
            print(f'Warning: {item} not found.')

    main(args, cfg, save_path, cp_path)
