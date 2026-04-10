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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.efficient import BUSISemiDataset, TwoStreamBatchSampler, mix_collate_fn
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import count_params, init_log, AverageMeter
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
}


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
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'])
    parser.add_argument('--base_lr', type=float, default=None)
    parser.add_argument('--conf_thresh', type=float, default=0.95)
    parser.add_argument('--cut_p', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--val_start', type=float, default=0.3)
    return parser


def load_cfg(args, defaults):
    cfg = dict(DEFAULT_BUSI_CFG)
    cfg.update(defaults)
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg.update(yaml.load(f, Loader=yaml.Loader))
    cfg['dataset'] = args.dataset
    return cfg


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(progress, args):
    return args.consistency * sigmoid_rampup(progress, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(buffer.data)


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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels, dropout_p))

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        ft_chns = params['feature_chns']
        dropout = params['dropout']
        self.in_conv = ConvBlock(params['in_chns'], ft_chns[0], dropout[0])
        self.down1 = DownBlock(ft_chns[0], ft_chns[1], dropout[1])
        self.down2 = DownBlock(ft_chns[1], ft_chns[2], dropout[2])
        self.down3 = DownBlock(ft_chns[2], ft_chns[3], dropout[3])
        self.down4 = DownBlock(ft_chns[3], ft_chns[4], dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class MLPOVRDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        ft_chns = params['feature_chns']
        n_class = params['class_num']
        self.up1 = UpBlock(ft_chns[4], ft_chns[3], ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(ft_chns[3], ft_chns[2], ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(ft_chns[2], ft_chns[1], ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(ft_chns[1], ft_chns[0], ft_chns[0], dropout_p=0.0)
        self.out_conv = nn.Conv2d(ft_chns[0], n_class, kernel_size=3, padding=1)
        self.projection_heads = nn.ModuleList()
        for _ in range(n_class - 1):
            self.projection_heads.append(
                nn.Sequential(
                    nn.Conv2d(ft_chns[0], ft_chns[0], kernel_size=1, padding=0),
                    nn.BatchNorm2d(ft_chns[0]),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(ft_chns[0], 3, kernel_size=3, padding=1),
                )
            )

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output, [head(x) for head in self.projection_heads]


class OVRUNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super().__init__()
        params = {
            'in_chns': in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': class_num,
            'bilinear': False,
        }
        self.encoder = Encoder(params)
        self.decoder = MLPOVRDecoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        return self.decoder(feature)


class SegmentationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


class MaskedDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target, mask=None):
        if target.ndim == inputs.ndim:
            target = target.squeeze(1)
        probs = torch.softmax(inputs, dim=1)
        target_onehot = F.one_hot(target.long(), num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        if mask is None:
            mask = torch.ones_like(target_onehot)
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[1] == 1:
                mask = mask.expand(-1, self.n_classes, -1, -1)
            mask = mask.float()
        probs = probs * mask
        target_onehot = target_onehot * mask
        intersection = (probs * target_onehot).sum(dim=(0, 2, 3))
        denominator = probs.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + 1e-10) / (denominator + 1e-10)
        return (1.0 - dice).mean()


def cross_entropy_masked(input_tensor, target, mask):
    loss_map = F.cross_entropy(input_tensor, target.long(), reduction='none')
    return (loss_map * mask.float()).mean()


def random_mask(batch_size, img_h, img_w, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3, device='cuda'):
    mask = torch.ones(batch_size, img_h, img_w, device=device)
    if random.random() > p:
        return mask
    for i in range(batch_size):
        size = np.random.uniform(size_min, size_max) * img_h * img_w
        while True:
            ratio = np.random.uniform(ratio_1, ratio_2)
            cutmix_w = int(np.sqrt(size / ratio))
            cutmix_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, max(img_w, 1))
            y = np.random.randint(0, max(img_h, 1))
            if x + cutmix_w <= img_w and y + cutmix_h <= img_h:
                break
        mask[i, y:y + cutmix_h, x:x + cutmix_w] = 0
    return mask


def build_class_names(cfg):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(str(cfg.get('dataset')).upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def strong_augment_labeled(img):
    scale = torch.empty(img.shape[0], 1, 1, 1, device=img.device).uniform_(0.9, 1.1)
    bias = torch.empty(img.shape[0], 1, 1, 1, device=img.device).uniform_(-0.05, 0.05)
    noise = torch.randn_like(img) * 0.03
    return img * scale + bias + noise


def generate_ovr_pseudo_label(pseudo, num_classes):
    pseudo_np = pseudo.detach().cpu().numpy()
    ovr_labels = []
    for i in range(num_classes - 1):
        current = np.zeros_like(pseudo_np, dtype=np.uint8)
        current[pseudo_np == i + 1] = 1
        for pix in range(1, num_classes):
            if pix != i + 1:
                current[pseudo_np == pix] = 2
        ovr_labels.append(torch.from_numpy(current).cuda())
    return ovr_labels


def diagnosis(outputs_ovr_soft):
    k = len(outputs_ovr_soft)
    pseudo = [torch.argmax(item, dim=1) for item in outputs_ovr_soft]
    pseudo = torch.stack(pseudo, dim=0)
    diag_matrix = torch.sum(pseudo, dim=0)
    return (diag_matrix == 0) | (diag_matrix == (2 * k - 1))


def generate_pseudo_label_withmask(outputs_ovr_soft, num_classes):
    background = torch.stack([item[:, 0, ...] for item in outputs_ovr_soft], dim=0).mean(dim=0)
    logits_map = [background]
    for item in outputs_ovr_soft:
        logits_map.append(item[:, 1, ...])
    logits_map = torch.stack(logits_map, dim=1)
    mask = diagnosis(outputs_ovr_soft)
    mask_with_channel = torch.stack([mask] * num_classes, dim=1)
    return logits_map.detach(), mask.detach(), mask_with_channel.detach()


def build_dataloaders(args, cfg):
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
    return trainloader, valloader


def build_optimizer(model, args, cfg):
    lr = args.base_lr if args.base_lr is not None else cfg['lr']
    if args.optimizer == 'AdamW':
        return AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)


def build_model(class_num):
    return init_2d_weight(OVRUNet(in_chns=3, class_num=class_num).cuda())


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model = build_model(cfg['nclass'])
    ema_model = build_model(cfg['nclass'])
    for param in ema_model.parameters():
        param.detach_()

    optimizer = build_optimizer(model, args, cfg)

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

    trainloader, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d', total_iters)
    class_names = build_class_names(cfg)
    log_interval = max(len(trainloader) // 8, 1)

    dice_loss = MaskedDiceLoss(cfg['nclass']).cuda()
    dice_loss_ovr = MaskedDiceLoss(3).cuda()

    best_performance = 0.0
    best_epoch = 0
    epoch = -1
    iter_num = 0
    latest_path = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_performance = checkpoint['best_performance']
        best_epoch = checkpoint['best_epoch']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        start_time = time.time()

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum: {args.labelnum}, '
            f'Previous best mdice ema: {best_performance:.4f} @epoch: {best_epoch}'
        )
        total_loss = AverageMeter()
        total_sup = AverageMeter()
        total_unsup = AverageMeter()
        total_mask_ratio = AverageMeter()
        model.train()
        ema_model.train()

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_u_w, img_u_s, ignore_mask, _ = unlabeled_data
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)
            img_u_w = img_u_w.cuda(non_blocking=True)
            img_u_s = img_u_s.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)

            num_lb = img_x.shape[0]
            num_ulb = img_u_w.shape[0]
            img_x_s = strong_augment_labeled(img_x)
            weak_batch = torch.cat((img_x, img_u_w), dim=0)
            strong_batch = torch.cat((img_x_s, img_u_s), dim=0)
            ovr_label = generate_ovr_pseudo_label(mask_x, cfg['nclass'])

            mix_perm = torch.randperm(num_ulb, device=img_u_w.device)
            img_u_w_mix = img_u_w[mix_perm]
            img_u_s_mix = img_u_s[mix_perm]
            weak_batch_mix = torch.cat((img_x, img_u_w_mix), dim=0)
            strong_batch_mix = torch.cat((img_x_s, img_u_s_mix), dim=0)

            img_mask = random_mask(weak_batch.shape[0], weak_batch.shape[2], weak_batch.shape[3], args.cut_p, device=weak_batch.device)
            img_mask_bool = img_mask.bool()
            img_mask_channel = img_mask.unsqueeze(1)

            with torch.no_grad():
                out_mix, out_mix_ovr = ema_model(weak_batch_mix)
                soft_mix = torch.softmax(out_mix, dim=1)
                pred_mix = torch.argmax(soft_mix[num_lb:].detach(), dim=1)
                mix_conf_mask = torch.max(soft_mix, dim=1)[0] > args.conf_thresh

                soft_mix_ovr = [torch.softmax(item, dim=1) for item in out_mix_ovr]
                pred_mix_ovr = [torch.argmax(item[num_lb:].detach(), dim=1) for item in soft_mix_ovr]
                mix_logits_ovr, mix_conf_mask_ovr, _ = generate_pseudo_label_withmask(soft_mix_ovr, cfg['nclass'])
                mix_pseudo_outputs_ovr = torch.argmax(mix_logits_ovr[num_lb:].detach(), dim=1)

            strong_batch[num_lb:] = img_mask_channel[num_lb:] * strong_batch[num_lb:] + (1 - img_mask_channel[num_lb:]) * strong_batch_mix[num_lb:]

            outputs_weak, outputs_weak_ovr = model(weak_batch)
            outputs_strong, outputs_strong_ovr = model(strong_batch)

            consistency_progress = iter_num / max(total_iters, 1) * args.consistency_rampup
            consistency_weight = get_current_consistency_weight(consistency_progress, args)

            sup_loss = F.cross_entropy(outputs_weak[:num_lb], mask_x.long()) + dice_loss(outputs_weak[:num_lb], mask_x)
            sup_loss_ovr = 0.0
            for cls_idx in range(cfg['nclass'] - 1):
                sup_loss_ovr += F.cross_entropy(outputs_weak_ovr[cls_idx][:num_lb], ovr_label[cls_idx].long())
                sup_loss_ovr += dice_loss_ovr(outputs_weak_ovr[cls_idx][:num_lb], ovr_label[cls_idx])
            sup_loss = sup_loss + sup_loss_ovr / max(cfg['nclass'] - 1, 1)

            with torch.no_grad():
                ema_output, ema_ovr_output = ema_model(weak_batch)
                ema_outputs_soft = torch.softmax(ema_output, dim=1)
                ema_ovr_output_soft = [torch.softmax(item, dim=1) for item in ema_ovr_output]

                conf_mask = torch.max(ema_outputs_soft, dim=1)[0] > args.conf_thresh
                conf_mask = torch.where(img_mask_bool, conf_mask, mix_conf_mask)
                mask_channel = conf_mask.unsqueeze(1).expand(-1, cfg['nclass'], -1, -1).float()

                pseudo_outputs = torch.argmax(ema_outputs_soft[num_lb:].detach(), dim=1)
                pseudo_outputs = torch.where(img_mask_bool[num_lb:], pseudo_outputs, pred_mix)
                pseudo_lab4ovr = generate_ovr_pseudo_label(pseudo_outputs, cfg['nclass'])

                pseudo_logits_ovr, pseudo_conf_mask_ovr, _ = generate_pseudo_label_withmask(ema_ovr_output_soft, cfg['nclass'])
                conf_mask_ovr = torch.where(img_mask_bool, pseudo_conf_mask_ovr, mix_conf_mask_ovr)
                mask_channel_ovr = conf_mask_ovr.unsqueeze(1).expand(-1, cfg['nclass'], -1, -1).float()

                pseudo_ovr = []
                for cls_idx in range(cfg['nclass'] - 1):
                    pred_ovr = torch.argmax(ema_ovr_output_soft[cls_idx][num_lb:].detach(), dim=1)
                    pseudo_ovr.append(torch.where(img_mask_bool[num_lb:], pred_ovr, pred_mix_ovr[cls_idx]))

                pseudo_outputs_ovr = torch.argmax(pseudo_logits_ovr[num_lb:].detach(), dim=1)
                pseudo_outputs_ovr = torch.where(img_mask_bool[num_lb:], pseudo_outputs_ovr, mix_pseudo_outputs_ovr)

                ensemble_mask = pseudo_outputs == pseudo_outputs_ovr
                ensemble_mask_channel = ensemble_mask.unsqueeze(1).expand(-1, cfg['nclass'], -1, -1).float()
                ensemble_mask_channel_ovr = ensemble_mask.unsqueeze(1).expand(-1, 3, -1, -1).float()
                ensemble_pseudo = pseudo_outputs * ensemble_mask.long()
                ensemble_pseudo4ovr = generate_ovr_pseudo_label(ensemble_pseudo, cfg['nclass'])
                mask_channel_bin = conf_mask.unsqueeze(1).expand(-1, 3, -1, -1).float()
                mask_channel_ovr_bin = conf_mask_ovr.unsqueeze(1).expand(-1, 3, -1, -1).float()

            unsup_loss = cross_entropy_masked(outputs_strong[num_lb:], pseudo_outputs.long(), conf_mask[num_lb:])
            unsup_loss = unsup_loss + dice_loss(outputs_strong[num_lb:], pseudo_outputs, mask=mask_channel[num_lb:])

            unsup_loss = unsup_loss + cross_entropy_masked(outputs_strong[num_lb:], pseudo_outputs_ovr.long(), conf_mask_ovr[num_lb:])
            unsup_loss = unsup_loss + dice_loss(outputs_strong[num_lb:], pseudo_outputs_ovr, mask=mask_channel_ovr[num_lb:])

            unsup_loss = unsup_loss + cross_entropy_masked(outputs_strong[num_lb:], ensemble_pseudo.long(), ensemble_mask)
            unsup_loss = unsup_loss + dice_loss(outputs_strong[num_lb:], ensemble_pseudo, mask=ensemble_mask_channel)

            unsup_loss_ovr = 0.0
            for cls_idx in range(cfg['nclass'] - 1):
                unsup_loss_ovr += cross_entropy_masked(outputs_strong_ovr[cls_idx][num_lb:], pseudo_lab4ovr[cls_idx].long(), conf_mask[num_lb:])
                unsup_loss_ovr += dice_loss_ovr(outputs_strong_ovr[cls_idx][num_lb:], pseudo_lab4ovr[cls_idx], mask=mask_channel_bin[num_lb:])

                unsup_loss_ovr += cross_entropy_masked(outputs_strong_ovr[cls_idx][num_lb:], pseudo_ovr[cls_idx].long(), conf_mask_ovr[num_lb:])
                unsup_loss_ovr += dice_loss_ovr(outputs_strong_ovr[cls_idx][num_lb:], pseudo_ovr[cls_idx], mask=mask_channel_ovr_bin[num_lb:])

                unsup_loss_ovr += cross_entropy_masked(outputs_strong_ovr[cls_idx][num_lb:], ensemble_pseudo4ovr[cls_idx].long(), ensemble_mask)
                unsup_loss_ovr += dice_loss_ovr(outputs_strong_ovr[cls_idx][num_lb:], ensemble_pseudo4ovr[cls_idx], mask=ensemble_mask_channel_ovr)

            unsup_loss = unsup_loss + unsup_loss_ovr / max(cfg['nclass'] - 1, 1)
            loss = sup_loss + consistency_weight * unsup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iters = epoch * len(trainloader) + i
            lr = (args.base_lr if args.base_lr is not None else cfg['lr']) * (1.0 - iters / max(total_iters, 1)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            iter_num += 1

            total_loss.update(loss.item())
            total_sup.update(sup_loss.item())
            total_unsup.update(unsup_loss.item())
            mask_ratio = conf_mask[num_lb:].float().mean().item() if num_ulb > 0 else 0.0
            total_mask_ratio.update(mask_ratio)

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_sup', sup_loss.item(), iters)
            writer.add_scalar('train/loss_unsup', unsup_loss.item(), iters)
            writer.add_scalar('train/consistency_weight', consistency_weight, iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {total_loss.avg:.3f}'
                    f', loss_sup: {total_sup.avg:.3f}, loss_unsup: {total_unsup.avg:.3f}, mask: {total_mask_ratio.avg:.4f}'
                )

        is_best = False
        if iter_num >= total_iters * args.val_start and epoch % args.val_interval == 0:
            ema_eval_model = SegmentationWrapper(ema_model)
            mDice, dice_class = eval_2d(valloader, ema_eval_model, cfg, ifdist=False, val_mode='ema')
            logger.info('*** Evaluation: MeanDice ema: {:.3f}'.format(mDice.item()))
            writer.add_scalar('eval/mDice_ema', mDice.item(), epoch)
            for cls_idx, dice in enumerate(dice_class):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info('*** Evaluation: Class [{:} {:}] Dice ema: {:.3f}'.format(cls_idx + 1, class_name, dice.item()))
                writer.add_scalar(f'eval/{class_name}_ema_DICE', dice.item(), epoch)
            if mDice.item() >= best_performance:
                best_performance = mDice.item()
                best_epoch = epoch
                is_best = True

        checkpoint = {
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_performance': best_performance,
            'best_epoch': best_epoch,
            'iter_num': iter_num,
            'start_time': start_time,
        }
        torch.save(checkpoint, latest_path)
        if is_best:
            model_ckpt = {'model': model.state_dict(), 'ema_model': ema_model.state_dict()}
            logger.info('*** best checkpoint: MeanDice ema: {:.3f}\n*** exp: {}'.format(best_performance, args.exp))
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_m{best_performance:.3f}.pth'))

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
        'CGS_BUSI/Ep{}bs{}_{}_seed{}_label{}/thresh{}_{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, args.conf_thresh, args.exp
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
