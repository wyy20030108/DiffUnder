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
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
from Datasets.efficient import ACDCsemiDataset
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
    parser.add_argument('--labelnum', type=int, default=cfgs['labelnum'], help=cfgs['label_help'])
    parser.add_argument('--num', default=cfgs['num'], type=int, help='unlabeled data number')
    parser.add_argument('--config', type=str, default=cfgs['config'])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=bool, default=False)
    parser.add_argument('--stage1_epochs', type=int, default=100)
    parser.add_argument('--dict_channels', type=int, default=512)
    parser.add_argument('--dict_lr_scale', type=float, default=1.0)
    parser.add_argument('--sparsity_weight', type=float, default=1e-4)
    parser.add_argument('--isoe_weight', type=float, default=0.2)
    parser.add_argument('--align_weight', type=float, default=1.0)
    parser.add_argument('--recon_weight', type=float, default=1.0)
    parser.add_argument('--cps_weight', type=float, default=1.0)
    parser.add_argument('--energy_consistency_weight', type=float, default=0.05)
    parser.add_argument('--clean_threshold', type=float, default=0.4)
    parser.add_argument('--ood_threshold', type=float, default=2.5)
    parser.add_argument('--reference_momentum', type=float, default=0.9)
    parser.add_argument('--conf_thresh', type=float, default=0.7)
    parser.add_argument('--conf_thresh_max', type=float, default=0.9)
    parser.add_argument('--min_mask_ratio', type=float, default=0.05)
    parser.add_argument('--energy_keep_ratio', type=float, default=0.7)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--val_start', type=float, default=0.3)
    return parser


class LabelConceptEncoder(nn.Module):
    def __init__(self, nclass, out_channels):
        super().__init__()
        hidden_channels = max(out_channels // 4, nclass)
        self.nclass = nclass
        self.proj = nn.Sequential(
            nn.Conv2d(nclass, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, mask, target_size):
        valid_mask = (mask >= 0) & (mask < self.nclass)
        safe_mask = mask.long().clone()
        safe_mask[~valid_mask] = 0
        one_hot = F.one_hot(safe_mask, num_classes=self.nclass).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid_mask.unsqueeze(1).float()
        one_hot = F.interpolate(one_hot, size=target_size, mode='nearest')
        return self.proj(one_hot)


class SparseConceptDictionary(nn.Module):
    def __init__(self, feature_channels, code_channels, clean_threshold=0.25, ood_threshold=0.75, momentum=0.9, energy_keep_ratio=0.7):
        super().__init__()
        self.feature_channels = feature_channels
        self.code_channels = code_channels
        self.clean_threshold = clean_threshold
        self.ood_threshold = ood_threshold
        self.momentum = momentum
        self.energy_keep_ratio = energy_keep_ratio
        self.encoder = nn.Conv2d(feature_channels, code_channels, kernel_size=1, bias=False)
        self.decoder = nn.Conv2d(code_channels, feature_channels, kernel_size=1, bias=False)
        self.register_buffer('reference_energy', torch.zeros(code_channels))
        self.register_buffer('alignment_error', torch.ones(code_channels))
        self.register_buffer('clean_mask', torch.ones(code_channels))
        self.register_buffer('reference_ready', torch.zeros(1))

    def encode(self, feature):
        pre_codes = self.encoder(feature)
        codes = F.relu(pre_codes)
        return pre_codes, codes

    def decode(self, codes):
        return self.decoder(codes)

    def forward(self, feature):
        pre_codes, codes = self.encode(feature)
        recon = self.decode(codes)
        return pre_codes, codes, recon

    def _channel_energy(self, codes):
        return codes.square().mean(dim=(0, 2, 3))

    def _sample_energy(self, codes, clean_mask):
        clean_idx = clean_mask.bool()
        if clean_idx.sum().item() == 0:
            clean_idx = torch.ones_like(clean_mask, dtype=torch.bool)
        return codes[:, clean_idx].square().mean(dim=(2, 3))

    def update_reference(self, image_codes, label_codes):
        with torch.no_grad():
            image_energy = self._channel_energy(image_codes)
            label_energy = self._channel_energy(label_codes)
            ref_energy = 0.5 * (image_energy + label_energy)
            rel_gap = (image_energy - label_energy).abs() / (ref_energy + 1e-6)
            if self.reference_ready.item() == 0:
                self.reference_energy.copy_(ref_energy)
                self.alignment_error.copy_(rel_gap)
                self.reference_ready.fill_(1)
            else:
                self.reference_energy.mul_(self.momentum).add_(ref_energy * (1.0 - self.momentum))
                self.alignment_error.mul_(self.momentum).add_(rel_gap * (1.0 - self.momentum))
            self.clean_mask.copy_((self.alignment_error <= self.clean_threshold).float())

    def get_clean_mask(self):
        clean_mask = self.clean_mask.bool()
        if clean_mask.sum().item() == 0:
            return torch.ones_like(clean_mask)
        return clean_mask

    def project(self, feature):
        pre_codes, codes = self.encode(feature)
        clean_mask = self.get_clean_mask()
        filtered_codes = codes * clean_mask.view(1, -1, 1, 1).float()
        recon = self.decode(filtered_codes)
        if self.reference_ready.item() == 1:
            clean_idx = clean_mask.bool()
            clean_codes = filtered_codes[:, clean_idx]
            ref_energy = self.reference_energy[clean_idx].view(1, -1, 1, 1).clamp_min(1e-6)
            local_deviation = (clean_codes.square() - ref_energy).abs() / ref_energy
            energy_score = local_deviation.mean(dim=1)
            adaptive_threshold = torch.quantile(
                energy_score.flatten(1),
                q=min(max(self.energy_keep_ratio, 0.0), 1.0),
                dim=1
            ).view(-1, 1, 1)
            energy_mask = energy_score <= torch.maximum(
                torch.full_like(adaptive_threshold, self.ood_threshold),
                adaptive_threshold
            )
        else:
            energy_mask = torch.ones(
                feature.shape[0], feature.shape[2], feature.shape[3], device=feature.device, dtype=torch.bool
            )
        return {
            'pre_codes': pre_codes,
            'codes': codes,
            'filtered_codes': filtered_codes,
            'recon_feature': recon,
            'energy_mask': energy_mask,
            'sample_energy': self._sample_energy(filtered_codes, clean_mask),
            'clean_mask': clean_mask,
        }


class DictionaryEvalWrapper(nn.Module):
    def __init__(self, backbone, dictionary):
        super().__init__()
        self.backbone = backbone
        self.dictionary = dictionary

    def forward(self, x):
        features = self.backbone.encoder(x)
        dict_out = self.dictionary.project(features[-1])
        features = list(features)
        features[-1] = dict_out['recon_feature']
        decoder_feature = self.backbone.decoder(features)
        return self.backbone.classifier(decoder_feature)


def build_class_names(dataset, nclass):
    class_names = CLASSES.get(dataset) or CLASSES.get(dataset.upper())
    if class_names is None:
        return [str(i) for i in range(1, nclass)]
    if len(class_names) == nclass:
        return class_names[1:]
    return class_names[:nclass - 1]


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


def encode_decode(backbone, x, bottleneck_override=None):
    features = backbone.encoder(x)
    bottleneck = features[-1]
    features = list(features)
    if bottleneck_override is not None:
        features[-1] = bottleneck_override
    decoder_feature = backbone.decoder(features)
    logits = backbone.classifier(decoder_feature)
    return logits, bottleneck, features


def energy_alignment_loss(image_codes, label_codes):
    image_energy = image_codes.square().mean(dim=(0, 2, 3))
    label_energy = label_codes.square().mean(dim=(0, 2, 3))
    return ((image_energy - label_energy).abs() / (image_energy + label_energy + 1e-6)).mean()


def code_alignment_loss(image_codes, label_codes):
    image_act = image_codes.abs().mean(dim=(2, 3))
    label_act = label_codes.abs().mean(dim=(2, 3))
    return F.l1_loss(image_act, label_act)


def augmentation_energy_loss(energy_a, energy_b):
    return ((energy_a - energy_b).abs() / (energy_a + energy_b + 1e-6)).mean()


def reconstruction_loss(x, x_hat):
    return F.mse_loss(x_hat, x)


def sparse_loss(codes):
    return codes.abs().mean()


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def build_loaders(args, cfg):
    trainset_l = ACDCsemiDataset('train_l', args, cfg['crop_size'])
    trainset_u = ACDCsemiDataset('train_u', args, cfg['crop_size'])
    valset = ACDCsemiDataset('val', args, cfg['crop_size'])
    loader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    loader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    return loader_l, loader_u, valloader


def make_supervised_loss(criterion, diceloss, logits, target):
    loss_ce = criterion(logits, target)
    valid_mask = (target >= 0) & (target < logits.shape[1])
    safe_target = target.clone()
    safe_target[~valid_mask] = 0
    loss_dice = diceloss(logits, safe_target)
    return (loss_ce + loss_dice) / 2.0, loss_ce, loss_dice


def masked_ce_loss(criterion, logits, pseudo, valid_mask):
    raw_loss = criterion(logits, pseudo)
    valid_mask = valid_mask.float()
    denom = valid_mask.sum().clamp_min(1.0)
    return (raw_loss * valid_mask).sum() / denom


def upscale_mask(mask, target_size):
    return F.interpolate(mask.unsqueeze(1).float(), size=target_size, mode='nearest').squeeze(1) > 0.5


def build_select_mask(confidence, valid_mask, energy_mask, base_thresh, min_ratio):
    candidate_mask = valid_mask & energy_mask
    select_mask = candidate_mask & (confidence >= base_thresh)
    if select_mask.any() or min_ratio <= 0:
        return select_mask

    batch_size = confidence.shape[0]
    refined_mask = torch.zeros_like(select_mask)
    for batch_idx in range(batch_size):
        candidate = candidate_mask[batch_idx]
        num_candidate = int(candidate.sum().item())
        if num_candidate == 0:
            refined_mask[batch_idx] = valid_mask[batch_idx] & (confidence[batch_idx] >= base_thresh)
            continue
        keep_count = max(1, int(num_candidate * min_ratio))
        candidate_scores = confidence[batch_idx][candidate]
        topk_scores = torch.topk(candidate_scores, k=min(keep_count, candidate_scores.numel()), dim=0).values
        adaptive_thresh = topk_scores.min()
        refined_mask[batch_idx] = candidate & (confidence[batch_idx] >= adaptive_thresh)
    return refined_mask


def stage1_forward(model1, model2, label_encoder, dictionary, img_x, mask_x, criterion_l, diceloss, args):
    logits1, bottleneck1, _ = encode_decode(model1, img_x)
    logits2, bottleneck2, _ = encode_decode(model2, img_x)
    label_feat = label_encoder(mask_x, bottleneck1.shape[-2:])
    _, img_codes1, img_recon1 = dictionary(bottleneck1)
    _, img_codes2, img_recon2 = dictionary(bottleneck2)
    _, lbl_codes, lbl_recon = dictionary(label_feat)
    dictionary.update_reference(0.5 * (img_codes1.detach() + img_codes2.detach()), lbl_codes.detach())
    sup1, loss_ce1, loss_dice1 = make_supervised_loss(criterion_l, diceloss, logits1, mask_x)
    sup2, loss_ce2, loss_dice2 = make_supervised_loss(criterion_l, diceloss, logits2, mask_x)
    dict_recon = (reconstruction_loss(bottleneck1, img_recon1) + reconstruction_loss(bottleneck2, img_recon2) + reconstruction_loss(label_feat, lbl_recon)) / 3.0
    dict_sparse = (sparse_loss(img_codes1) + sparse_loss(img_codes2) + sparse_loss(lbl_codes)) / 3.0
    dict_isoe = 0.5 * (energy_alignment_loss(img_codes1, lbl_codes) + energy_alignment_loss(img_codes2, lbl_codes))
    dict_align = 0.5 * (code_alignment_loss(img_codes1, lbl_codes) + code_alignment_loss(img_codes2, lbl_codes))
    dict_loss = (
        args.recon_weight * dict_recon
        + args.sparsity_weight * dict_sparse
        + args.isoe_weight * dict_isoe
        + args.align_weight * dict_align
    )
    total_loss = sup1 + sup2 + dict_loss
    return {
        'loss': total_loss,
        'sup1': sup1,
        'sup2': sup2,
        'ce1': loss_ce1,
        'ce2': loss_ce2,
        'dice1': loss_dice1,
        'dice2': loss_dice2,
        'dict_loss': dict_loss,
        'dict_recon': dict_recon,
        'dict_sparse': dict_sparse,
        'dict_isoe': dict_isoe,
        'dict_align': dict_align,
        'clean_ratio': dictionary.get_clean_mask().float().mean(),
    }


def stage2_forward(model1, model2, label_encoder, dictionary, img_x, mask_x, img_u_w, img_u_s, ignore_mask, criterion_l, criterion_u, diceloss, args, stage2_progress):
    stage1_stats = stage1_forward(model1, model2, label_encoder, dictionary, img_x, mask_x, criterion_l, diceloss, args)

    _, bottleneck1_w, _ = encode_decode(model1, img_u_w)
    _, bottleneck1_s, _ = encode_decode(model1, img_u_s)
    _, bottleneck2_w, _ = encode_decode(model2, img_u_w)
    _, bottleneck2_s, _ = encode_decode(model2, img_u_s)

    dict1_w = dictionary.project(bottleneck1_w)
    dict1_s = dictionary.project(bottleneck1_s)
    dict2_w = dictionary.project(bottleneck2_w)
    dict2_s = dictionary.project(bottleneck2_s)

    logits1_w, _, _ = encode_decode(model1, img_u_w, bottleneck_override=dict1_w['recon_feature'])
    logits1_s, _, _ = encode_decode(model1, img_u_s, bottleneck_override=dict1_s['recon_feature'])
    logits2_w, _, _ = encode_decode(model2, img_u_w, bottleneck_override=dict2_w['recon_feature'])
    logits2_s, _, _ = encode_decode(model2, img_u_s, bottleneck_override=dict2_s['recon_feature'])

    prob1 = torch.softmax(logits1_w.detach(), dim=1)
    prob2 = torch.softmax(logits2_w.detach(), dim=1)
    conf1, pseudo1 = prob1.max(dim=1)
    conf2, pseudo2 = prob2.max(dim=1)
    current_conf_thresh = args.conf_thresh + (args.conf_thresh_max - args.conf_thresh) * stage2_progress

    energy_mask1 = upscale_mask(dict1_w['energy_mask'], logits1_w.shape[-2:])
    energy_mask2 = upscale_mask(dict2_w['energy_mask'], logits2_w.shape[-2:])
    valid_mask = ignore_mask != 255
    select_mask1 = build_select_mask(conf1, valid_mask, energy_mask1, current_conf_thresh, args.min_mask_ratio)
    select_mask2 = build_select_mask(conf2, valid_mask, energy_mask2, current_conf_thresh, args.min_mask_ratio)

    cps_loss_1 = masked_ce_loss(criterion_u, logits1_s, pseudo2, select_mask2)
    cps_loss_2 = masked_ce_loss(criterion_u, logits2_s, pseudo1, select_mask1)
    cps_loss = 0.5 * (cps_loss_1 + cps_loss_2)

    energy_consistency = 0.5 * (
        augmentation_energy_loss(dict1_w['sample_energy'], dict1_s['sample_energy'])
        + augmentation_energy_loss(dict2_w['sample_energy'], dict2_s['sample_energy'])
    )

    unlabeled_recon = 0.25 * (
        reconstruction_loss(bottleneck1_w, dict1_w['recon_feature'])
        + reconstruction_loss(bottleneck1_s, dict1_s['recon_feature'])
        + reconstruction_loss(bottleneck2_w, dict2_w['recon_feature'])
        + reconstruction_loss(bottleneck2_s, dict2_s['recon_feature'])
    )

    total_loss = (
        stage1_stats['sup1']
        + stage1_stats['sup2']
        + stage1_stats['dict_loss']
        + args.cps_weight * cps_loss
        + args.energy_consistency_weight * energy_consistency
        + 0.1 * unlabeled_recon
    )

    stage1_stats.update({
        'loss': total_loss,
        'cps_loss': cps_loss,
        'energy_consistency': energy_consistency,
        'unlabeled_recon': unlabeled_recon,
        'mask_ratio_1': select_mask1.float().mean(),
        'mask_ratio_2': select_mask2.float().mean(),
        'conf_thresh': torch.tensor(current_conf_thresh, device=img_x.device),
    })
    return stage1_stats


def evaluate(model, dictionary, valloader, cfg, name):
    wrapper = DictionaryEvalWrapper(model, dictionary)
    return eval_2d(valloader, wrapper, cfg, ifdist=False, val_mode=name)


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model1 = init_2d_weight(UNet(in_chns=1, class_num=cfg['nclass']).cuda()).cuda()
    model2 = init_2d_weight(UNet(in_chns=1, class_num=cfg['nclass']).cuda()).cuda()
    label_encoder = LabelConceptEncoder(cfg['nclass'], 256).cuda()
    dictionary = SparseConceptDictionary(
        feature_channels=256,
        code_channels=args.dict_channels,
        clean_threshold=args.clean_threshold,
        ood_threshold=args.ood_threshold,
        momentum=args.reference_momentum,
        energy_keep_ratio=args.energy_keep_ratio,
    ).cuda()

    param_groups = [
        {'params': model1.parameters(), 'lr': cfg['lr'], 'lr_scale': 1.0},
        {'params': model2.parameters(), 'lr': cfg['lr'], 'lr_scale': 1.0},
        {'params': label_encoder.parameters(), 'lr': cfg['lr'] * args.dict_lr_scale, 'lr_scale': args.dict_lr_scale},
        {'params': dictionary.parameters(), 'lr': cfg['lr'] * args.dict_lr_scale, 'lr_scale': args.dict_lr_scale},
    ]
    optimizer = AdamW(param_groups, lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()

    loader_l, loader_u, valloader = build_loaders(args, cfg)
    stage1_steps = len(loader_l)
    stage2_steps = max(len(loader_l), len(loader_u))
    total_epochs = args.stage1_epochs + cfg['epochs']
    total_iters = args.stage1_epochs * stage1_steps + cfg['epochs'] * stage2_steps
    class_names = build_class_names(cfg['dataset'], cfg['nclass'])

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(
        count_params(model1) + count_params(model2) + count_params(label_encoder) + count_params(dictionary)
    ))
    logger.info('Total iters: %d' % total_iters)

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
        label_encoder.load_state_dict(checkpoint['label_encoder'])
        dictionary.load_state_dict(checkpoint['dictionary'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        pre_best_dice1 = checkpoint['previous_best1']
        pre_best_dice2 = checkpoint['previous_best2']
        best_epoch1 = checkpoint['best_epoch1']
        best_epoch2 = checkpoint['best_epoch2']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    log_interval = max(stage2_steps // 8, 1)
    labeled_iter = infinite_loader(loader_l)
    unlabeled_iter = infinite_loader(loader_u)

    for epoch in range(epoch + 1, total_epochs):
        is_stage1 = epoch < args.stage1_epochs
        stage_name = 'stage1' if is_stage1 else 'stage2'
        logger.info(
            f'===> Epoch: {epoch}/{total_epochs}, stage:{stage_name}, seed:{args.seed}, labelnum:{args.labelnum}, '
            f'best1:{pre_best_dice1:.4f}@{best_epoch1}, best2:{pre_best_dice2:.4f}@{best_epoch2}'
        )
        meters = {
            'loss': AverageMeter(),
            'sup1': AverageMeter(),
            'sup2': AverageMeter(),
            'dict_loss': AverageMeter(),
            'dict_isoe': AverageMeter(),
            'dict_align': AverageMeter(),
            'clean_ratio': AverageMeter(),
            'cps_loss': AverageMeter(),
            'energy_consistency': AverageMeter(),
            'mask_ratio_1': AverageMeter(),
            'mask_ratio_2': AverageMeter(),
            'conf_thresh': AverageMeter(),
        }
        model1.train()
        model2.train()
        label_encoder.train()
        dictionary.train()

        steps_this_epoch = stage1_steps if is_stage1 else stage2_steps
        for step in range(steps_this_epoch):
            img_x, mask_x = next(labeled_iter)
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)

            if is_stage1:
                stats = stage1_forward(model1, model2, label_encoder, dictionary, img_x, mask_x, criterion_l, diceloss, args)
            else:
                img_u_w, img_u_s, ignore_mask, _ = next(unlabeled_iter)
                img_u_w = img_u_w.cuda(non_blocking=True)
                img_u_s = img_u_s.cuda(non_blocking=True)
                ignore_mask = ignore_mask.cuda(non_blocking=True)
                stage2_progress = min(max((epoch - args.stage1_epochs) / max(cfg['epochs'] - 1, 1), 0.0), 1.0)
                stats = stage2_forward(
                    model1, model2, label_encoder, dictionary,
                    img_x, mask_x, img_u_w, img_u_s, ignore_mask,
                    criterion_l, criterion_u, diceloss, args, stage2_progress
                )

            optimizer.zero_grad()
            stats['loss'].backward()
            optimizer.step()

            current_iter = iter_num
            lr = cfg['lr'] * (1 - current_iter / max(total_iters, 1)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
            iter_num += 1
            global_step = iter_num

            for key, meter in meters.items():
                if key in stats:
                    meter.update(float(stats[key].item() if torch.is_tensor(stats[key]) else stats[key]))
                    writer.add_scalar(f'train/{stage_name}_{key}', meter.val, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

            if step % log_interval == 0:
                logger.info(
                    'Iters: {:}/{:}, stage: {}, loss: {:.3f}, sup1: {:.3f}, sup2: {:.3f}, dict: {:.3f}, cps: {:.3f}, energy: {:.3f}, clean: {:.3f}, mask1: {:.3f}, mask2: {:.3f}'.format(
                        global_step,
                        total_iters,
                        stage_name,
                        meters['loss'].avg,
                        meters['sup1'].avg,
                        meters['sup2'].avg,
                        meters['dict_loss'].avg,
                        meters['cps_loss'].avg,
                        meters['energy_consistency'].avg,
                        meters['clean_ratio'].avg,
                        meters['mask_ratio_1'].avg,
                        meters['mask_ratio_2'].avg,
                    )
                )

        should_validate = (epoch + 1) >= args.stage1_epochs and iter_num >= total_iters * args.val_start and epoch % args.val_interval == 0
        is_best = False
        if should_validate:
            mDice1, dice_class1 = evaluate(model1, dictionary, valloader, cfg, 'IESeg_model1')
            mDice2, dice_class2 = evaluate(model2, dictionary, valloader, cfg, 'IESeg_model2')
            for cls_idx, dice in enumerate(dice_class1):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info('*** Evaluation: Class [{:} {:}] Dice model1: {:.3f}, model2: {:.3f}'.format(
                    cls_idx + 1, class_name, dice.item(), dice_class2[cls_idx].item()
                ))
                writer.add_scalar(f'eval/{class_name}_model1_DICE', dice.item(), epoch)
                writer.add_scalar(f'eval/{class_name}_model2_DICE', dice_class2[cls_idx].item(), epoch)
            writer.add_scalar('eval/mDice_model1', mDice1.item(), epoch)
            writer.add_scalar('eval/mDice_model2', mDice2.item(), epoch)
            logger.info('*** Evaluation: MeanDice model1: {:.3f}, model2: {:.3f}'.format(mDice1.item(), mDice2.item()))
            if mDice1.item() >= pre_best_dice1:
                pre_best_dice1 = mDice1.item()
                best_epoch1 = epoch
                torch.save({'model': model1.state_dict(), 'dictionary': dictionary.state_dict()}, os.path.join(cp_path, f'best_model1_ep{epoch}_m{pre_best_dice1:.3f}.pth'))
                is_best = True
            if mDice2.item() >= pre_best_dice2:
                pre_best_dice2 = mDice2.item()
                best_epoch2 = epoch
                torch.save({'model': model2.state_dict(), 'dictionary': dictionary.state_dict()}, os.path.join(cp_path, f'best_model2_ep{epoch}_m{pre_best_dice2:.3f}.pth'))
                is_best = True

        checkpoint = {
            'model1': model1.state_dict(),
            'model2': model2.state_dict(),
            'label_encoder': label_encoder.state_dict(),
            'dictionary': dictionary.state_dict(),
            'optimizer': optimizer.state_dict(),
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
            logger.info('*** best checkpoint updated: model1 {:.3f}@{}, model2 {:.3f}@{}'.format(
                pre_best_dice1, best_epoch1, pre_best_dice2, best_epoch2
            ))

    end_time = time.time()
    logger.info('Training time: {:.2f}s'.format(end_time - start_time))
    writer.close()
    gc.collect()
    torch.cuda.empty_cache()


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
        'IESeg/Ep{}bs{}_{}_seed{}_label{}/stage1{}_{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, args.stage1_epochs, args.exp
        ),
    )
    os.makedirs(cp_path, exist_ok=True)
    save_path = os.path.join(cp_path, 'log')
    os.makedirs(save_path, exist_ok=True)

    include_list = ['comparison', 'utils', 'configs', 'Datasets', 'models', 'scripts', 'tools', 'IESeg.py', 'readme.md']
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
