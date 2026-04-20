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
from Datasets.efficient import ACDCsemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import AverageMeter, DiceLoss, count_params, init_log
from utils.val import eval_2d


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def get_parser(datasetname):
    cfgs = DATASET_CONFIGS[datasetname]
    parser = argparse.ArgumentParser(description=datasetname)
    parser.add_argument('--dataset', type=str, default=datasetname, choices=DATASET_CONFIGS.keys())
    parser.add_argument('--base_dir', type=str, default=cfgs['base_dir'])
    parser.add_argument('--labelnum', type=int, default=cfgs['labelnum'], help=cfgs['label_help'])
    parser.add_argument('--num', default=cfgs['num'], type=int, help='unlabeled data number')
    parser.add_argument('--config', type=str, default=cfgs['config'])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:7')
    parser.add_argument('--exp', type=str, help='Experiment description')
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
    return parser


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
    w = np.random.randint(img_size[1] / ratio + 1, img_size[1] + 1)
    h = int(np.round(cutout_area / w))
    h = max(1, min(h, img_size[0]))
    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)
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


def get_revisiting_loss(random_pool, rep_u, rep_u_teacher, topk=5):
    rep_u = F.adaptive_avg_pool2d(rep_u, 1).flatten(1)
    rep_u = F.normalize(rep_u, dim=-1)
    rep_u_teacher = F.adaptive_avg_pool2d(rep_u_teacher, 1).flatten(1)
    rep_u_teacher = F.normalize(rep_u_teacher, dim=-1)
    dist_t = 2 - 2 * torch.einsum('bc,kc->bk', rep_u, random_pool)
    dist_q = 2 - 2 * torch.einsum('bc,kc->bk', rep_u_teacher, random_pool)
    topk = min(topk, random_pool.shape[0])
    _, nn_index = dist_t.topk(topk, dim=1, largest=False)
    nn_dist_q = torch.gather(dist_q, 1, nn_index)
    return (nn_dist_q.sum(dim=1) / topk).mean(), rep_u_teacher.detach()


def monte_carlo_sample(high, shape):
    if high <= 0:
        return torch.zeros((shape,), dtype=torch.long)
    return torch.randint(high, size=(shape,), dtype=torch.long)


def grid_monte_carlo_sample(high, shape, cut_count=4):
    if high <= 0:
        return torch.zeros((shape,), dtype=torch.long)
    try:
        edge = round(math.sqrt(high))
        if edge <= 1:
            return monte_carlo_sample(high, shape)
        samples = []
        per_patch_sample = max(1, shape * edge * edge // max(high, 1) // (cut_count ** 2))
        img = np.arange(edge * edge).reshape(edge, edge)
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
                picked = picked[np.random.permutation(len(picked))]
                indices = np.random.randint(0, len(picked), size=(per_patch_sample,))
                samples.append(picked[indices])
        sample_tensor = torch.as_tensor(np.concatenate(samples), dtype=torch.long)
        sample_tensor = sample_tensor[sample_tensor < high]
        if sample_tensor.numel() < shape:
            pad = torch.randint(high, (shape - sample_tensor.numel(),), dtype=torch.long)
            sample_tensor = torch.cat((sample_tensor, pad), dim=0)
        return sample_tensor[torch.randperm(sample_tensor.numel())][:shape]
    except Exception:
        return monte_carlo_sample(high, shape)


def grid_as_monte_carlo_sample(high, shape, cut_count=4):
    if high <= 0:
        return torch.zeros((shape,), dtype=torch.long)
    try:
        edge = round(math.sqrt(high))
        if edge <= 1:
            return monte_carlo_sample(high, shape)
        samples = []
        per_patch_sample = max(2, shape * edge * edge // max(high, 1) // (cut_count ** 2))
        img = np.arange(edge * edge).reshape(edge, edge)
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
                picked = picked[np.random.permutation(len(picked))]
                center = int(2 * np.mean(picked))
                half = max(1, per_patch_sample // 2)
                indices = np.random.randint(0, len(picked), size=(half,))
                chosen = picked[indices]
                mirrored = center - chosen
                samples.extend([chosen, mirrored])
        sample_tensor = torch.as_tensor(np.concatenate(samples), dtype=torch.long)
        sample_tensor = sample_tensor[(sample_tensor >= 0) & (sample_tensor < high)]
        if sample_tensor.numel() < shape:
            pad = torch.randint(high, (shape - sample_tensor.numel(),), dtype=torch.long)
            sample_tensor = torch.cat((sample_tensor, pad), dim=0)
        return sample_tensor[torch.randperm(sample_tensor.numel())][:shape]
    except Exception:
        return monte_carlo_sample(high, shape)


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


def compute_contra_memobank_loss(
    rep,
    label_l,
    label_u,
    prob_l,
    prob_u,
    low_mask,
    high_mask,
    memobank,
    queue_ptrlis,
    queue_size,
    rep_teacher,
    delta_n=1.0,
    func='smc',
    num_queries=256,
    num_negatives=512,
    temp=0.5,
):
    num_feat = rep.shape[1]
    num_segments = label_l.shape[1]
    low_rank = min(3, num_segments)
    high_rank = max(low_rank + 1, min(20, num_segments))
    sample_fn = {'asmc': grid_as_monte_carlo_sample, 'smc': grid_monte_carlo_sample}.get(func, monte_carlo_sample)

    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    _, prob_indices_l = torch.sort(prob_l, 1, descending=True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)
    _, prob_indices_u = torch.sort(prob_u, 1, descending=True)
    prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)
    prob = torch.cat((prob_l, prob_u), dim=0)

    class_entries = []
    for cls_idx in range(num_segments):
        low_valid_pixel_seg = low_valid_pixel[:, cls_idx].bool()
        high_valid_pixel_seg = high_valid_pixel[:, cls_idx].bool()
        prob_seg = prob[:, cls_idx]

        rep_mask_low_entropy = (prob_seg > 0.3) & low_valid_pixel_seg
        rep_mask_high_entropy = (prob_seg < delta_n) & high_valid_pixel_seg
        seg_feat_low_entropy = rep[rep_mask_low_entropy]

        if high_rank > low_rank:
            class_mask_u = prob_indices_u[:, :, :, low_rank:high_rank].eq(cls_idx).any(dim=3)
        else:
            class_mask_u = prob_indices_u[:, :, :, -1:].eq(cls_idx).any(dim=3)
        class_mask_l = prob_indices_l[:, :, :, :max(low_rank, 1)].eq(cls_idx).any(dim=3)
        class_mask = torch.cat((class_mask_l & (label_l[:, cls_idx] == 0), class_mask_u), dim=0)
        negative_mask = rep_mask_high_entropy & class_mask
        enqueue_memobank(rep_teacher[negative_mask], memobank[cls_idx], queue_ptrlis[cls_idx], queue_size[cls_idx])

        if low_valid_pixel_seg.any():
            seg_proto = rep_teacher[low_valid_pixel_seg].detach().mean(dim=0, keepdim=True)
            class_entries.append((cls_idx, seg_proto, seg_feat_low_entropy))

    if len(class_entries) <= 1:
        return rep.sum() * 0.0

    reco_loss = rep.new_tensor(0.0)
    valid_count = 0
    for cls_idx, seg_proto, seg_feat_low_entropy in class_entries:
        bank = memobank[cls_idx][0]
        if seg_feat_low_entropy.shape[0] == 0 or bank.shape[0] == 0:
            continue

        anchor_idx = sample_fn(seg_feat_low_entropy.shape[0], num_queries).to(seg_feat_low_entropy.device)
        anchor_feat = seg_feat_low_entropy[anchor_idx]

        negative_idx = sample_fn(bank.shape[0], num_queries * num_negatives)
        negative_feat = bank[negative_idx].to(anchor_feat.device).reshape(num_queries, num_negatives, num_feat)
        positive_feat = seg_proto.unsqueeze(0).repeat(num_queries, 1, 1).to(anchor_feat.device)
        all_feat = torch.cat((positive_feat, negative_feat), dim=1)

        seg_logits = F.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
        reco_loss = reco_loss + F.cross_entropy(
            seg_logits / temp,
            torch.zeros(num_queries, dtype=torch.long, device=anchor_feat.device),
        )
        valid_count += 1

    if valid_count == 0:
        return rep.sum() * 0.0
    return reco_loss / valid_count


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model = UNet(in_chns=1, class_num=cfg['nclass']).cuda()
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
    # optimizer = AdamW(
    #     params=model.parameters(),
    #     lr=cfg['lr'],
    #     betas=(0.9, 0.999),
    #     weight_decay=0.01,
    # )

    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()

    num_gpus = torch.cuda.device_count()
    logger.info('use {} gpus!'.format(num_gpus))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

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

    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d', total_iters)

    random_pool = F.normalize(torch.randn(args.K, feature_dim_sum, device='cuda'), dim=1)
    random_pool_ptr = torch.zeros(1, dtype=torch.long, device='cuda')
    memobank = [[torch.empty((0, feature_dim_sum), dtype=torch.float32)] for _ in range(cfg['nclass'])]
    queue_ptrlis = [torch.zeros(1, dtype=torch.long) for _ in range(cfg['nclass'])]
    queue_size = [args.memobank_bg_size] + [args.memobank_size] * (cfg['nclass'] - 1)

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
        reco_meter = AverageMeter()
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
            img_x, mask_x = labeled_data
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)

            img_w, img_s, ignore_mask, _ = unlabeled_data
            img_w = img_w.cuda(non_blocking=True)
            img_s = img_s.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)

            with torch.no_grad():
                pred_u_teacher_weak, _ = forward_with_features(ema_model, img_w)
                pseudo_prob_weak = torch.softmax(pred_u_teacher_weak, dim=1)
                pseudo_logits, pseudo_u = torch.max(pseudo_prob_weak, dim=1)
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = generate_unsup_data(
                    img_s,
                    pseudo_u.clone(),
                    pseudo_logits.clone(),
                    mode=args.apply_aug,
                )

            pred_l, l_feature_map = forward_with_features(model, img_x)
            pred_u, u_feature_map = forward_with_features(model, train_u_aug_data)

            with torch.no_grad():
                pred_l_teacher, l_feature_map_teacher = forward_with_features(ema_model, img_x)
                pred_u_teacher, u_feature_map_teacher = forward_with_features(ema_model, train_u_aug_data)

            l_feature_all = q_feature_extractor(l_feature_map)
            u_feature_all = q_feature_extractor(u_feature_map)
            l_feature_all_teacher = k_feature_extractor(l_feature_map_teacher)
            u_feature_all_teacher = k_feature_extractor(u_feature_map_teacher)

            rep_l = q_representation(l_feature_all)
            rep_u = q_representation(u_feature_all)
            rep_l_teacher = l_feature_all_teacher
            rep_u_teacher = u_feature_all_teacher

            loss_ce = criterion_l(pred_l, mask_x)
            loss_dice = diceloss(pred_l, mask_x)
            supervised_loss = loss_ce + loss_dice

            train_u_aug_ignore_mask = ignore_mask
            unsup_loss = compute_unsupervised_loss(
                pred_u,
                train_u_aug_label,
                train_u_aug_logits,
                args.strong_threshold,
                ignore_mask=train_u_aug_ignore_mask,
            )

            valid_u_mask = (train_u_aug_label >= 0) & (train_u_aug_ignore_mask != 255)
            alpha_t = 20 * (1 - epoch / max(cfg['epochs'], 1))
            with torch.no_grad():
                label_l = label_onehot(mask_x, cfg['nclass'])
                label_u = label_onehot(train_u_aug_label, cfg['nclass'])
                prob_l_teacher = torch.softmax(pred_l_teacher, dim=1)
                prob_u_teacher = torch.softmax(pred_u_teacher, dim=1)

                entropy = -torch.sum(prob_u_teacher * torch.log(prob_u_teacher + 1e-10), dim=1)
                valid_entropy = entropy[valid_u_mask]
                low_thresh = percentile_threshold(valid_entropy, alpha_t, 0.0)
                high_thresh = percentile_threshold(valid_entropy, 100 - alpha_t, 1.0)
                low_entropy_mask = entropy.le(low_thresh) & valid_u_mask
                high_entropy_mask = entropy.ge(high_thresh) & valid_u_mask

                low_mask_all = torch.cat(
                    (
                        torch.ones((mask_x.shape[0], 1, mask_x.shape[1], mask_x.shape[2]), device=mask_x.device),
                        low_entropy_mask.unsqueeze(1).float(),
                    ),
                    dim=0,
                )
                high_mask_all = torch.cat(
                    (
                        torch.ones((mask_x.shape[0], 1, mask_x.shape[1], mask_x.shape[2]), device=mask_x.device),
                        high_entropy_mask.unsqueeze(1).float(),
                    ),
                    dim=0,
                )

            reco_loss = compute_contra_memobank_loss(
                rep=torch.cat((rep_l, rep_u), dim=0),
                label_l=label_l,
                label_u=label_u,
                prob_l=prob_l_teacher.detach(),
                prob_u=prob_u_teacher.detach(),
                low_mask=low_mask_all,
                high_mask=high_mask_all,
                memobank=memobank,
                queue_ptrlis=queue_ptrlis,
                queue_size=queue_size,
                rep_teacher=torch.cat((rep_l_teacher, rep_u_teacher), dim=0).detach(),
                delta_n=args.strong_threshold_u2pl,
                func=args.func,
                num_queries=args.num_queries,
                num_negatives=args.num_negatives,
                temp=args.reco_temperature,
            )

            loss_q, pooled_rep_u_teacher = get_revisiting_loss(random_pool, rep_u, rep_u_teacher, topk=args.topk)
            dequeue_and_enqueue(pooled_rep_u_teacher, random_pool, random_pool_ptr)

            loss = (
                supervised_loss
                + args.unsup_weight * unsup_loss
                + args.reco_weight * reco_loss
                + args.revisit_weight * loss_q
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            reco_meter.update(reco_loss.item())
            revisit_meter.update(loss_q.item())

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/loss_sup', supervised_loss.item(), iters)
            writer.add_scalar('train/loss_unsup', unsup_loss.item(), iters)
            writer.add_scalar('train/loss_reco', reco_loss.item(), iters)
            writer.add_scalar('train/loss_revisit', loss_q.item(), iters)
            writer.add_scalar('train/pseudo_fg_ratio', (train_u_aug_label > 0).float().mean().item(), iters)
            writer.add_scalar('train/pseudo_conf_ratio', train_u_aug_logits.ge(args.strong_threshold).float().mean().item(), iters)

            if i % max(args.num // 2, 1) == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {loss_meter.avg:.3f}, '
                    f'sup: {sup_meter.avg:.3f}, unsup: {unsup_meter.avg:.3f}, '
                    f'reco: {reco_meter.avg:.3f}, revisit: {revisit_meter.avg:.3f}, l_q: {loss_q.item():.3f}'
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
                class_name = CLASSES[cfg['dataset']][cls_idx]
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
    data_parser.add_argument('--cli_dataset', type=str)
    known_args, remaining_args = data_parser.parse_known_args()
    dataset_name = known_args.cli_dataset
    print(dataset_name)
    if dataset_name not in DATASET_CONFIGS:
        print(f'Error: {dataset_name} not found in configs.')
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
    with open(args.config, 'r') as f:
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
