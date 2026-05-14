import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')

import argparse
import gc
import logging
import math
import pprint
import random
import re
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Datasets.efficient import BUSISemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import AverageMeter, DiceLoss, count_params, init_log
from utils.val import eval_2d


ABLATION_PRESETS = {
    'A1': {
        'name': 'ours_full',
        'description': 'Ours (Full): UOT + single prototype + adaptive running prior/relax',
        'overrides': {
            'enable_unsup': True,
            'enable_uot': True,
            'enable_proto': True,
            'proto_from_uot': True,
            'enable_adaptive_relax': True,
            'prior_mode': 'running',
            'num_prototypes_per_class': 1,
            'apply_aug': 'cutmix',
        },
    },
    'A2': {
        'name': 'ours_wo_uot',
        'description': 'Ours w/o UOT: use raw teacher probabilities for pseudo labels and prototype extraction',
        'overrides': {
            'enable_unsup': True,
            'enable_uot': False,
            'enable_proto': True,
            'proto_from_uot': False,
            'num_prototypes_per_class': 1,
        },
    },
    'A3': {
        'name': 'baseline_plus_uot',
        'description': 'Baseline + UOT: keep UOT-aligned consistency but remove prototype learning',
        'overrides': {
            'enable_unsup': True,
            'enable_uot': True,
            'enable_proto': False,
            'proto_from_uot': True,
            'num_prototypes_per_class': 1,
        },
    },
    'A4': {
        'name': 'baseline_plus_pseudo_label',
        'description': 'Baseline + Pseudo Label: raw teacher probabilities without UOT or prototypes',
        'overrides': {
            'enable_unsup': True,
            'enable_uot': False,
            'enable_proto': False,
            'proto_from_uot': False,
            'num_prototypes_per_class': 1,
        },
    },
    'A5': {
        'name': 'baseline_supervised_only',
        'description': 'Baseline (Supervised Only): remove all unlabeled objectives',
        'overrides': {
            'enable_unsup': False,
            'enable_uot': False,
            'enable_proto': False,
            'proto_from_uot': False,
            'num_prototypes_per_class': 1,
        },
    },
    'A6': {
        'name': 'fixed_relax',
        'description': 'Fixed Relax: disable adaptive UOT relax scheduling',
        'overrides': {
            'enable_unsup': True,
            'enable_uot': True,
            'enable_proto': True,
            'proto_from_uot': True,
            'enable_adaptive_relax': False,
            'prior_mode': 'running',
            'num_prototypes_per_class': 1,
        },
    },
    'A7': {
        'name': 'fixed_prior',
        'description': 'Fixed Prior: replace running prior with fixed uniform prior',
        'overrides': {
            'enable_unsup': True,
            'enable_uot': True,
            'enable_proto': True,
            'proto_from_uot': True,
            'enable_adaptive_relax': True,
            'prior_mode': 'fixed_uniform',
            'num_prototypes_per_class': 1,
        },
    },
    'A8': {
        'name': 'multi_proto_k4',
        'description': 'Multi-Proto (K=4): keep full setting but use four prototypes per foreground class',
        'overrides': {
            'enable_unsup': True,
            'enable_uot': True,
            'enable_proto': True,
            'proto_from_uot': True,
            'enable_adaptive_relax': True,
            'prior_mode': 'running',
            'num_prototypes_per_class': 4,
        },
    },
    'A9': {
        'name': 'no_cutmix',
        'description': 'No CutMix: disable unlabeled mixing augmentation in the full setting',
        'overrides': {
            'apply_aug': 'none',
            'num_prototypes_per_class': 1,
        },
    },
}


ABALATION_HELP_TEXT = """ 时间开销比较, 显存资源开销
A1  ours_full          : UOT + single prototype + adaptive running prior/relax      # 0.732 Dice: 0.8643, Jaccard: 0.7664, HD95: 2.4794, ASD: 0.9643
A2  ours_wo_uot        : remove UOT, use raw teacher pseudo labels for prototypes   # 0.624 Dice: 0.7580, Jaccard: 0.6698, HD95: 6.8912, ASD: 1.5918
A3  baseline_plus_uot  : supervised + UOT consistency only                          # 0.756 Dice: 0.8714, Jaccard: 0.7780, HD95: 2.2620, ASD: 0.7431
A4  baseline_plus_pseudo_label: supervised + raw pseudo label consistency only      # 0.717 Dice: 0.8420, Jaccard: 0.7495, HD95: 7.3334, ASD: 1.9399
A5  baseline_supervised_only : supervised only                                      # 0.445 Dice: 0.4801, Jaccard: 0.3591, HD95: 56.7545, ASD: 22.4312
A6  fixed_relax        : full setting with fixed relax                              # 0.768 Dice: 0.8748, Jaccard: 0.7825, HD95: 2.2665, ASD: 0.6191 --- Dice: 0.7867, Jaccard: 0.6907, HD95: 7.1125, ASD: 1.7424
A7  fixed_prior        : full setting with fixed uniform prior                      # 0.182 Dice: 0.1439, Jaccard: 0.0827, HD95: 93.2875, ASD: 46.4250
A8  multi_proto_k4     : full setting with K=4 prototypes                           # 0.706 Dice: 0.8312, Jaccard: 0.7348, HD95: 3.9165, ASD: 1.4769
A9  no_cutmix          : full setting without CutMix                                # 0.679 Dice: 0.8030, Jaccard: 0.7001, HD95: 10.5442, ASD: 3.8658
""".strip()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def print_ablation_table():
    print('Available ablation presets:')
    print(ABALATION_HELP_TEXT)


def apply_ablation_preset(args):
    preset = ABLATION_PRESETS[args.ablation_id]
    for key, value in preset['overrides'].items():
        setattr(args, key, value)
    args.ablation_name = preset['name']
    args.ablation_desc = preset['description']
    if not args.exp:
        args.exp = f'{args.ablation_id}_{args.ablation_name}'
    return args


def get_parser(datasetname):
    cfgs = DATASET_CONFIGS[datasetname]
    parser = argparse.ArgumentParser(description=datasetname)
    parser.add_argument('--dataset', type=str, default=datasetname, choices=DATASET_CONFIGS.keys())
    parser.add_argument('--base_dir', type=str, default=cfgs['base_dir'])
    parser.add_argument('--labelnum', type=int, default=cfgs['labelnum'], help=cfgs['label_help'])
    parser.add_argument('--num', default=cfgs['num'], type=int, help='unlabeled data number')
    parser.add_argument('--config', type=str, default=cfgs['config'])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp', type=str, default=None, help='Optional experiment alias')
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.99)

    parser.add_argument('--ablation_id', type=str, default='A1', choices=sorted(ABLATION_PRESETS.keys()))
    parser.add_argument('--list_ablations', action='store_true', default=False)
    parser.add_argument('--ablation_name', type=str, default=None)
    parser.add_argument('--ablation_desc', type=str, default=None)

    parser.add_argument('--enable_unsup', type=str2bool, default=True)
    parser.add_argument('--enable_uot', type=str2bool, default=True)
    parser.add_argument('--enable_proto', type=str2bool, default=True)
    parser.add_argument('--proto_from_uot', type=str2bool, default=True)
    parser.add_argument('--enable_sharpen', type=str2bool, default=True)
    parser.add_argument('--enable_adaptive_relax', type=str2bool, default=True)

    parser.add_argument('--strong_threshold', type=float, default=0.97)
    parser.add_argument('--apply_aug', type=str, default='cutmix', choices=['cutout', 'cutmix', 'classmix', 'none'])
    parser.add_argument('--func', type=str, default='smc', choices=['asmc', 'smc', 'rand'])
    parser.add_argument('--num_queries', type=int, default=256)

    parser.add_argument('--unsup_weight', type=float, default=1.0)
    parser.add_argument('--proto_weight', type=float, default=0.01)

    parser.add_argument('--sinkhorn_eps', type=float, default=0.5)
    parser.add_argument('--sinkhorn_iters', type=int, default=5)
    parser.add_argument('--sinkhorn_relax', type=float, default=10.0)
    parser.add_argument('--sinkhorn_relax_min', type=float, default=0.1)
    parser.add_argument('--sinkhorn_fg_prob_threshold', type=float, default=0.05)
    parser.add_argument('--profile_time_overhead', type=str2bool, default=True)
    parser.add_argument('--prior_mode', type=str, default='running', choices=['running', 'fixed_uniform', 'fixed_custom'])
    parser.add_argument('--class_prior', type=str, default=None)
    parser.add_argument('--prior_momentum', type=float, default=0.99)

    parser.add_argument('--prototype_momentum', type=float, default=0.99)
    parser.add_argument('--prototype_threshold', type=float, default=0.8)
    parser.add_argument('--prototype_min_threshold', type=float, default=0.5)
    parser.add_argument('--prototype_temperature', type=float, default=0.5)
    parser.add_argument('--num_prototypes_per_class', type=int, default=1)
    parser.add_argument('--sharpen_temperature', type=float, default=0.5)

    parser.add_argument('--auto_test_after_train', type=str2bool, default=True)
    parser.add_argument('--auto_test_split', type=str, default='test', choices=['test'])
    parser.add_argument('--auto_test_save_pred_root', type=str, default=None)
    parser.add_argument('--auto_update_readme', type=str2bool, default=True)
    parser.add_argument('--readme_path', type=str, default=os.path.join(project_root, 'readme.md'))
    parser.add_argument('--readme_method', type=str, default=None)
    return parser


def infer_method_name_from_cp_path(cp_path, checkpoint_root, dataset, fallback='Ours'):
    try:
        relative_cp_path = os.path.relpath(cp_path, checkpoint_root)
        method_segment = relative_cp_path.split(os.sep)[0]
    except ValueError:
        method_segment = os.path.basename(os.path.dirname(cp_path))

    method_name = (method_segment or '').strip()
    dataset_tokens = {str(dataset), str(dataset).upper(), str(dataset).lower()}
    for token in sorted(dataset_tokens, key=len, reverse=True):
        suffix = f'_{token}'
        if method_name.endswith(suffix):
            method_name = method_name[:-len(suffix)]
            break
    method_name = method_name.strip('_- ')
    return method_name or fallback


def parse_scored_checkpoint_name(file_name):
    match = re.fullmatch(r'ep(?P<epoch>\d+)_m(?P<model>\d+(?:\.\d+)?)_ema(?P<ema>\d+(?:\.\d+)?)\.pth', file_name)
    if match is None:
        return None
    return {
        'epoch': int(match.group('epoch')),
        'model_score': float(match.group('model')),
        'ema_score': float(match.group('ema')),
    }


def select_best_inference_checkpoint(cp_path, prefer_ema=False):
    stable_name = 'best_ema.pth' if prefer_ema else 'best_model.pth'
    stable_path = os.path.join(cp_path, stable_name)
    if os.path.exists(stable_path):
        return stable_path, prefer_ema, stable_name

    scored_candidates = []
    for file_name in os.listdir(cp_path):
        parsed = parse_scored_checkpoint_name(file_name)
        if parsed is None:
            continue
        parsed['path'] = os.path.join(cp_path, file_name)
        scored_candidates.append(parsed)

    if scored_candidates:
        score_key = 'ema_score' if prefer_ema else 'model_score'
        best_candidate = max(scored_candidates, key=lambda item: (item[score_key], item['epoch']))
        return best_candidate['path'], prefer_ema, 'scored_history'

    latest_ckpt_path = os.path.join(cp_path, 'latest.pth')
    if os.path.exists(latest_ckpt_path):
        return latest_ckpt_path, False, 'latest'
    raise FileNotFoundError(f'Cannot find any checkpoint under {cp_path}')


def run_post_training_busi_inference(args, cfg, cp_path, logger, best_model_score, best_ema_score):
    from test_busi import build_fixed_cfg as build_eval_cfg
    from test_busi import test_busi as run_test_busi
    from test_busi import update_readme_results

    prefer_ema = best_ema_score >= best_model_score
    best_ckpt_path, use_ema, checkpoint_source = select_best_inference_checkpoint(cp_path, prefer_ema=prefer_ema)
    method_name = args.readme_method or infer_method_name_from_cp_path(
        cp_path=cp_path,
        checkpoint_root=args.checkpoint_path,
        dataset=cfg['dataset'],
        fallback='Ours',
    )

    eval_args = argparse.Namespace(
        dataset=cfg['dataset'],
        base_dir=args.base_dir,
        labelnum=args.labelnum,
        num=args.num,
        device=args.device,
        seed=args.seed,
        split=args.auto_test_split,
        save_model_path=best_ckpt_path,
        use_ema=use_ema,
        nclass=cfg['nclass'],
        crop_size=cfg.get('crop_size'),
        strict_eval_masks=False,
        save_pred=False,
        save_pred_root=args.auto_test_save_pred_root,
        readme_path=args.readme_path,
        readme_method=method_name,
        skip_readme_update=True,
    )
    eval_cfg = build_eval_cfg(eval_args)

    logger.info(
        'Auto inference checkpoint: %s, use_ema=%s, source=%s, best_model=%.4f, best_ema=%.4f, method=%s',
        best_ckpt_path,
        use_ema,
        checkpoint_source,
        best_model_score,
        best_ema_score,
        method_name,
    )
    avg_metrics, mean_metrics = run_test_busi(eval_args, eval_cfg)

    if args.auto_update_readme:
        update_readme_results(
            readme_path=args.readme_path,
            dataset=cfg['dataset'],
            seed=args.seed,
            labelnum=args.labelnum,
            method_name=method_name,
            avg_metrics=avg_metrics,
            mean_metrics=mean_metrics,
        )
        logger.info('Updated README %s -> %s', args.readme_path, method_name)

    return {
        'checkpoint_path': best_ckpt_path,
        'use_ema': use_ema,
        'method_name': method_name,
        'avg_metrics': avg_metrics,
        'mean_metrics': mean_metrics,
    }


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


def split_feature_maps(feature_maps, split_sizes):
    split_per_level = [torch.split(feature, split_sizes, dim=0) for feature in feature_maps]
    return tuple([level_splits[idx] for level_splits in split_per_level] for idx in range(len(split_sizes)))


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
    uniform_prior = torch.full((num_classes,), 1.0 / num_classes, device=device)
    if args.prior_mode == 'fixed_uniform':
        return uniform_prior, True

    prior_source = args.class_prior if args.class_prior is not None else cfg.get('class_prior')
    if args.prior_mode == 'fixed_custom':
        if prior_source is None:
            raise ValueError('prior_mode=fixed_custom requires --class_prior or cfg["class_prior"].')
        return parse_class_prior(prior_source, num_classes).to(device), True

    if prior_source is None:
        return uniform_prior, False
    return parse_class_prior(prior_source, num_classes).to(device), False


@torch.no_grad()
def update_running_prior(running_prior, labels, num_classes, momentum):
    valid_labels = labels[(labels >= 0) & (labels != 255)]
    if valid_labels.numel() == 0:
        return
    if valid_labels.max() == 0: 
        return
    batch_hist = torch.bincount(valid_labels.view(-1), minlength=num_classes).float().to(running_prior.device)
    batch_hist = normalize_distribution(batch_hist)
    running_prior.mul_(momentum).add_(batch_hist, alpha=1 - momentum)
    running_prior.copy_(normalize_distribution(running_prior))


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


def build_zero_loss(reference_tensor):
    return reference_tensor.sum() * 0.0


def sync_cuda_for_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def start_wall_timer(enabled):
    if not enabled:
        return None
    sync_cuda_for_timing()
    return time.perf_counter()


def stop_wall_timer(start_time, enabled):
    if not enabled or start_time is None:
        return 0.0
    sync_cuda_for_timing()
    return time.perf_counter() - start_time


def main(args, cfg, save_path, cp_path):
    exp_tag = args.exp
    logger = init_log('global', logging.INFO, os.path.join(save_path, exp_tag))
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
    q_representation = RepresentationHead(feature_dim_sum, feature_dim_sum).cuda()
    k_feature_extractor = deepcopy(q_feature_extractor).cuda()
    for param in k_feature_extractor.parameters():
        param.detach_()

    trainable_params = []
    for module in (model, q_feature_extractor, q_representation):
        trainable_params.extend(param for param in module.parameters() if param.requires_grad)
    optimizer = AdamW(
        params=trainable_params,
        lr=cfg['lr'],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()

    num_gpus = torch.cuda.device_count()
    logger.info('use {} gpus!'.format(num_gpus))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))
    logger.info('Ablation: %s -> %s', args.ablation_id, args.ablation_desc)

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
    total_iter_time_meter = AverageMeter()
    total_fixmatch_time_meter = AverageMeter()
    total_overhead_time_meter = AverageMeter()
    total_uot_time_meter = AverageMeter()
    total_repr_time_meter = AverageMeter()
    total_proto_time_meter = AverageMeter()

    latest_ckpt_path = os.path.join(cp_path, 'latest.pth')
    best_model_ckpt_path = os.path.join(cp_path, 'best_model.pth')
    best_ema_ckpt_path = os.path.join(cp_path, 'best_ema.pth')
    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        q_feature_extractor.load_state_dict(checkpoint['q_feature_extractor'])
        k_feature_extractor.load_state_dict(checkpoint['k_feature_extractor'])
        q_representation.load_state_dict(checkpoint['q_representation'])
        optimizer.load_state_dict(checkpoint['optimizer'])
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
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, {exp_tag}, seed:{args.seed}, labelnum:{args.labelnum}, '
            f'Previous best mdice model: {pre_best_dice:.4f} @epoch: {best_epoch}, '
            f'ema: {pre_best_dice_ema:.4f} @epoch_ema: {best_epoch_ema}'
        )

        loss_meter = AverageMeter()
        sup_meter = AverageMeter()
        unsup_meter = AverageMeter()
        proto_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        fixmatch_time_meter = AverageMeter()
        overhead_time_meter = AverageMeter()
        uot_time_meter = AverageMeter()
        repr_time_meter = AverageMeter()
        proto_time_meter = AverageMeter()
        model_improved = False
        ema_improved = False
        is_best = False
        mDice = torch.tensor(pre_best_dice)
        mDice_ema = torch.tensor(pre_best_dice_ema)

        model.train()
        ema_model.train()
        q_feature_extractor.train()
        k_feature_extractor.train()
        q_representation.train()

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            iter_start_time = start_wall_timer(args.profile_time_overhead)
            extra_time_this_iter = 0.0

            img_x, mask_x = labeled_data
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)

            img_w, img_s, ignore_mask, _ = unlabeled_data
            img_w = img_w.cuda(non_blocking=True)
            img_s = img_s.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)

            labeled_batch_size = img_x.shape[0]
            unlabeled_batch_size = img_w.shape[0]

            with torch.no_grad():
                if args.enable_proto:
                    teacher_inputs = torch.cat((img_w, img_x), dim=0)
                    teacher_logits, teacher_feature_maps = forward_with_features(ema_model, teacher_inputs)
                    pred_u_teacher_weak, _ = torch.split(
                        teacher_logits,
                        [unlabeled_batch_size, labeled_batch_size],
                        dim=0,
                    )
                    u_feature_map_teacher_weak, l_feature_map_teacher = split_feature_maps(
                        teacher_feature_maps,
                        [unlabeled_batch_size, labeled_batch_size],
                    )
                else:
                    pred_u_teacher_weak = ema_model(img_w)
                uot_start_time = start_wall_timer(args.profile_time_overhead)
                if not use_fixed_prior:
                    update_running_prior(target_prior, mask_x, cfg['nclass'], args.prior_momentum)

                weak_valid_u_mask = ignore_mask != 255
                raw_prob_u_weak = torch.softmax(pred_u_teacher_weak, dim=1)
                if args.enable_uot:
                    if args.enable_adaptive_relax:
                        current_relax = compute_adaptive_sinkhorn_relax(
                            prob_map=raw_prob_u_weak.detach(),
                            valid_mask=weak_valid_u_mask,
                            base_relax=args.sinkhorn_relax,
                            min_relax=args.sinkhorn_relax_min,
                            fg_prob_threshold=args.sinkhorn_fg_prob_threshold,
                        )
                    else:
                        current_relax = max(float(args.sinkhorn_relax), 1e-3)
                    soft_prob_u_weak = sinkhorn_align_distribution(
                        raw_prob_u_weak.detach(),
                        weak_valid_u_mask,
                        target_prior,
                        eps=args.sinkhorn_eps,
                        num_iters=args.sinkhorn_iters,
                        relax=current_relax,
                    )
                else:
                    current_relax = 0.0
                    soft_prob_u_weak = raw_prob_u_weak.detach()
                uot_time = stop_wall_timer(uot_start_time, args.profile_time_overhead)
                extra_time_this_iter += uot_time

                pseudo_conf_u_weak, pseudo_label_u_weak = torch.max(soft_prob_u_weak, dim=1)
                train_u_aug_data, train_u_aug_label, train_u_aug_logits, train_u_aug_soft_label, train_u_aug_ignore_mask = generate_unsup_data_with_soft_targets(
                    img_s,
                    pseudo_label_u_weak.clone(),
                    pseudo_conf_u_weak.clone(),
                    soft_prob_u_weak.clone(),
                    ignore_mask.clone(),
                    mode=args.apply_aug,
                )

            student_inputs = torch.cat((img_x, train_u_aug_data), dim=0)
            if args.enable_proto:
                student_logits, student_feature_maps = forward_with_features(model, student_inputs)
            else:
                student_logits = model(student_inputs)
            pred_l, pred_u = torch.split(
                student_logits,
                [labeled_batch_size, train_u_aug_data.shape[0]],
                dim=0,
            )
            if args.enable_proto:
                l_feature_map, u_feature_map = split_feature_maps(
                    student_feature_maps,
                    [labeled_batch_size, train_u_aug_data.shape[0]],
                )

                repr_start_time = start_wall_timer(args.profile_time_overhead)
                student_feature_all = q_feature_extractor([
                    torch.cat((l_feat, u_feat), dim=0) for l_feat, u_feat in zip(l_feature_map, u_feature_map)
                ])
                teacher_feature_all = k_feature_extractor([
                    torch.cat((l_feat, u_feat), dim=0) for l_feat, u_feat in zip(l_feature_map_teacher, u_feature_map_teacher_weak)
                ])
                l_feature_all_teacher, u_feature_all_teacher = torch.split(
                    teacher_feature_all,
                    [labeled_batch_size, unlabeled_batch_size],
                    dim=0,
                )

                rep_all = q_representation(student_feature_all)
                rep_l, rep_u = torch.split(
                    rep_all,
                    [labeled_batch_size, train_u_aug_data.shape[0]],
                    dim=0,
                )
                repr_time = stop_wall_timer(repr_start_time, args.profile_time_overhead)
                extra_time_this_iter += repr_time
            else:
                repr_time = 0.0

            loss_ce = criterion_l(pred_l, mask_x)
            loss_dice = diceloss(pred_l, mask_x)
            supervised_loss = loss_ce + loss_dice

            if args.enable_sharpen:
                train_u_aug_soft_label = sharpen_probability_map(
                    train_u_aug_soft_label.detach(),
                    temperature=args.sharpen_temperature,
                )
            else:
                train_u_aug_soft_label = train_u_aug_soft_label.detach()

            valid_u_mask = (
                (train_u_aug_label >= 0)
                & (train_u_aug_ignore_mask != 255)
                & train_u_aug_logits.ge(args.strong_threshold)
            )

            if args.enable_unsup:
                unsup_loss = compute_distribution_alignment_loss(
                    pred_u,
                    train_u_aug_soft_label,
                    valid_u_mask,
                )
            else:
                unsup_loss = build_zero_loss(pred_u)

            if args.enable_proto:
                proto_start_time = start_wall_timer(args.profile_time_overhead)
                with torch.no_grad():
                    if args.proto_from_uot:
                        proto_conf_u_weak = pseudo_conf_u_weak
                        proto_label_u_weak = pseudo_label_u_weak
                    else:
                        proto_conf_u_weak, proto_label_u_weak = torch.max(raw_prob_u_weak.detach(), dim=1)

                    prototype_valid_u_weak = build_classwise_prototype_mask(
                        proto_conf_u_weak,
                        proto_label_u_weak,
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
                        torch.cat((l_feature_all_teacher, u_feature_all_teacher), dim=0).detach(),
                        torch.cat((mask_x, proto_label_u_weak), dim=0),
                        torch.cat((labeled_valid, prototype_valid_u_weak), dim=0),
                        args.prototype_momentum,
                        num_prototypes_per_class=args.num_prototypes_per_class,
                        assignment_temp=args.prototype_temperature,
                    )

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
                proto_time = stop_wall_timer(proto_start_time, args.profile_time_overhead)
                extra_time_this_iter += proto_time
            else:
                proto_loss = build_zero_loss(pred_u)
                proto_time = 0.0

            loss = (
                supervised_loss
                + args.unsup_weight * unsup_loss
                + args.proto_weight * proto_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            if args.enable_proto:
                momentum_update_module(q_feature_extractor, k_feature_extractor, args.ema_decay)

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            loss_meter.update(loss.item())
            sup_meter.update(supervised_loss.item())
            unsup_meter.update(unsup_loss.item())
            proto_meter.update(proto_loss.item())

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/loss_sup', supervised_loss.item(), iters)
            writer.add_scalar('train/loss_unsup', unsup_loss.item(), iters)
            writer.add_scalar('train/loss_proto', proto_loss.item(), iters)
            writer.add_scalar('train/pseudo_fg_ratio', (train_u_aug_label > 0).float().mean().item(), iters)
            writer.add_scalar('train/pseudo_conf_ratio', train_u_aug_logits.ge(args.strong_threshold).float().mean().item(), iters)
            writer.add_scalar('train/pseudo_conf_valid_ratio', valid_u_mask.float().mean().item(), iters)
            writer.add_scalar('train/prior_fg_ratio', target_prior[1:].sum().item(), iters)
            writer.add_scalar('train/uot_relax', float(current_relax), iters)
            if args.profile_time_overhead:
                iter_time = stop_wall_timer(iter_start_time, args.profile_time_overhead)
                iter_time_ms = iter_time * 1000.0
                overhead_time_ms = min(extra_time_this_iter, iter_time) * 1000.0
                fixmatch_time_ms = max(iter_time - extra_time_this_iter, 0.0) * 1000.0
                iter_time_meter.update(iter_time_ms)
                fixmatch_time_meter.update(fixmatch_time_ms)
                overhead_time_meter.update(overhead_time_ms)
                uot_time_meter.update(uot_time * 1000.0)
                repr_time_meter.update(repr_time * 1000.0)
                proto_time_meter.update(proto_time * 1000.0)
                total_iter_time_meter.update(iter_time_ms)
                total_fixmatch_time_meter.update(fixmatch_time_ms)
                total_overhead_time_meter.update(overhead_time_ms)
                total_uot_time_meter.update(uot_time * 1000.0)
                total_repr_time_meter.update(repr_time * 1000.0)
                total_proto_time_meter.update(proto_time * 1000.0)
                writer.add_scalar('time/iter_ms', iter_time_ms, iters)
                writer.add_scalar('time/fixmatch_iter_ms', fixmatch_time_ms, iters)
                writer.add_scalar('time/overhead_iter_ms', overhead_time_ms, iters)
                writer.add_scalar('time/uot_iter_ms', uot_time * 1000.0, iters)
                writer.add_scalar('time/repr_iter_ms', repr_time * 1000.0, iters)
                writer.add_scalar('time/proto_iter_ms', proto_time * 1000.0, iters)
                if fixmatch_time_ms > 1e-8:
                    writer.add_scalar('time/overhead_ratio_pct', overhead_time_ms / fixmatch_time_ms * 100.0, iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {loss_meter.avg:.3f}, '
                    f'sup: {sup_meter.avg:.3f}, unsup: {unsup_meter.avg:.3f}, '
                    f'proto: {proto_meter.avg:.3f}, relax: {float(current_relax):.3f}'
                )

        if args.profile_time_overhead and iter_time_meter.count > 0:
            overhead_ratio_pct = 0.0
            if fixmatch_time_meter.avg > 1e-8:
                overhead_ratio_pct = overhead_time_meter.avg / fixmatch_time_meter.avg * 100.0
            logger.info(
                'Epoch %d timing | total: %.2f ms/iter | fixmatch: %.2f ms/iter | overhead: %.2f ms/iter (%.2f%%) '
                '| uot: %.2f | repr: %.2f | proto: %.2f',
                epoch,
                iter_time_meter.avg,
                fixmatch_time_meter.avg,
                overhead_time_meter.avg,
                overhead_ratio_pct,
                uot_time_meter.avg,
                repr_time_meter.avg,
                proto_time_meter.avg,
            )
            writer.add_scalar('time_epoch/iter_ms', iter_time_meter.avg, epoch)
            writer.add_scalar('time_epoch/fixmatch_iter_ms', fixmatch_time_meter.avg, epoch)
            writer.add_scalar('time_epoch/overhead_iter_ms', overhead_time_meter.avg, epoch)
            writer.add_scalar('time_epoch/uot_iter_ms', uot_time_meter.avg, epoch)
            writer.add_scalar('time_epoch/repr_iter_ms', repr_time_meter.avg, epoch)
            writer.add_scalar('time_epoch/proto_iter_ms', proto_time_meter.avg, epoch)
            writer.add_scalar('time_epoch/overhead_ratio_pct', overhead_ratio_pct, epoch)

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

            model_improved = mDice.item() >= pre_best_dice
            ema_improved = mDice_ema.item() >= pre_best_dice_ema
            is_best = model_improved or ema_improved
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
            'class_prototypes': class_prototypes.detach().cpu(),
            'prototype_mask': prototype_mask.detach().cpu(),
            'target_prior': target_prior.detach().cpu(),
        }
        if is_best:
            logger.info(
                '*** best checkpoint: MeanDice model: {:.3f}, ema: {:.3f}\n*** exp: {}'.format(
                    mDice.item(),
                    mDice_ema.item(),
                    exp_tag,
                )
            )
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_m{mDice.item():.3f}_ema{mDice_ema.item():.3f}.pth'))
        if model_improved:
            torch.save(model_ckpt, best_model_ckpt_path)
        if ema_improved:
            torch.save(model_ckpt, best_ema_ckpt_path)

    end_time = time.time()
    logger.info('Training time: %.2fs', end_time - start_time)
    if args.profile_time_overhead and total_iter_time_meter.count > 0:
        total_overhead_ratio_pct = 0.0
        if total_fixmatch_time_meter.avg > 1e-8:
            total_overhead_ratio_pct = total_overhead_time_meter.avg / total_fixmatch_time_meter.avg * 100.0
        logger.info(
            'Training timing summary | total: %.2f ms/iter | fixmatch: %.2f ms/iter | overhead: %.2f ms/iter (%.2f%%) '
            '| uot: %.2f | repr: %.2f | proto: %.2f',
            total_iter_time_meter.avg,
            total_fixmatch_time_meter.avg,
            total_overhead_time_meter.avg,
            total_overhead_ratio_pct,
            total_uot_time_meter.avg,
            total_repr_time_meter.avg,
            total_proto_time_meter.avg,
        )
    writer.close()
    gc.collect()
    torch.cuda.empty_cache()

    if args.auto_test_after_train:
        run_post_training_busi_inference(
            args=args,
            cfg=cfg,
            cp_path=cp_path,
            logger=logger,
            best_model_score=pre_best_dice,
            best_ema_score=pre_best_dice_ema,
        )


if __name__ == '__main__':
    data_parser = argparse.ArgumentParser(description='datasets')
    data_parser.add_argument('--cli_dataset', type=str)
    data_parser.add_argument('--list_ablations', action='store_true', default=False)
    known_args, remaining_args = data_parser.parse_known_args()

    if known_args.list_ablations:
        print_ablation_table()
        sys.exit(0)

    dataset_name = known_args.cli_dataset
    print(dataset_name)
    if dataset_name not in DATASET_CONFIGS:
        print(f'Error: {dataset_name} not found in configs.')
        sys.exit(1)

    parser = get_parser(dataset_name)
    args = parser.parse_args(remaining_args)
    if args.list_ablations:
        print_ablation_table()
        sys.exit(0)
    args = apply_ablation_preset(args)

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
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path) # project_root 是你在文件头部定义好的
        
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    cp_path = os.path.join(
        args.checkpoint_path,
        'OT_ARCO/Ep{}bs{}_{}_seed{}_label{}/{}'.format(
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

    include_list = ['comparsion', 'utils', 'configs', 'Datasets', 'models', 'scripts', 'tools']
    target_dir = os.path.join(cp_path, 'code')
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    for item in include_list:
        src = os.path.join(project_root, item)
        dst = os.path.join(target_dir, item)
        if os.path.exists(src):
            if os.path.isdir(src):
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        else:
            print(f'Warning: {item} not found.')

    main(args, cfg, save_path, cp_path)
