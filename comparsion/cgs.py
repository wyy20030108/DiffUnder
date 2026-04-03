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
import re
import h5py
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import count_params, init_log, AverageMeter
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
    parser.add_argument('--model', type=str, default='OVRUNet')
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--conf_thresh', type=float, default=0.95)
    parser.add_argument('--cut_p', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    parser.add_argument('--labeled_bs', type=int, default=None)
    parser.add_argument('--unlabeled_bs', type=int, default=None)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--val_start', type=float, default=0.5)
    return parser


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
        if inputs.shape[1] != self.n_classes:
            raise ValueError('inputs channel number does not match n_classes')
        if target.ndim == inputs.ndim:
            target = target.squeeze(1)
        probs = torch.softmax(inputs, dim=1) if inputs.dtype.is_floating_point else inputs
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
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2.0 * intersection + 1e-10) / (denominator + 1e-10)
        return (1.0 - dice).mean()


def cross_entropy_masked(input_tensor, target, mask):
    loss_map = F.cross_entropy(input_tensor, target.long(), reduction='none')
    mask = mask.float()
    return (loss_map * mask).mean()


def random_mask(batch_size, img_h, img_w, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
    mask = torch.ones(batch_size, img_h, img_w, device='cuda')
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


def build_class_names(cfg, args):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(args.dataset) or CLASSES.get(args.dataset.upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


class CGSACDCDataset(Dataset):
    def __init__(self, mode, args, crop_size, num_classes):
        self.mode = mode
        self.args = args
        self.crop_size = self._normalize_size(crop_size)
        self.num_classes = num_classes
        self.root_path = args.base_dir
        self.images_h5_dir = self._resolve_images_h5_dir()
        self.all_slice_names = self._list_all_slice_names()
        self.patient_to_slices = self._build_patient_to_slices(self.all_slice_names)
        if mode == 'train':
            self.labeled_names, self.unlabeled_names = self._build_train_split()
            self.sample_list = self.labeled_names + self.unlabeled_names
            self.labeled_count = len(self.labeled_names)
        else:
            self.sample_list = self._build_eval_list(mode)

    def _normalize_size(self, crop_size):
        if isinstance(crop_size, int):
            return (crop_size, crop_size)
        if isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
            return (int(crop_size[-2]), int(crop_size[-1]))
        raise ValueError('crop_size must be int, list or tuple')

    def _resolve_images_h5_dir(self):
        for folder in ['Images_h5', 'images_h5', 'data/slices', 'data']:
            folder_path = os.path.join(self.root_path, folder)
            if os.path.isdir(folder_path):
                return folder_path
        raise FileNotFoundError(f'Cannot find image h5 directory under {self.root_path}')

    def _list_all_slice_names(self):
        return sorted([file_name[:-3] for file_name in os.listdir(self.images_h5_dir) if file_name.endswith('.h5')])

    def _extract_patient_id(self, sample_name):
        match = re.search(r'(patient\d+)', sample_name)
        if match is None:
            raise ValueError(f'Cannot parse patient id from {sample_name}')
        return match.group(1)

    def _build_patient_to_slices(self, slice_names):
        mapping = {}
        for sample_name in slice_names:
            patient_id = self._extract_patient_id(sample_name)
            mapping.setdefault(patient_id, []).append(sample_name)
        for patient_id in mapping:
            mapping[patient_id] = sorted(mapping[patient_id])
        return mapping

    def _resolve_split_file(self, mode):
        candidates = {
            'train': ['train.txt', 'train.list', 'train_slices.list'],
            'val': ['val.txt', 'val.list'],
            'test': ['test.txt', 'test.list'],
        }[mode]
        for candidate in candidates:
            split_path = os.path.join(self.root_path, candidate)
            if os.path.exists(split_path):
                return split_path
        return None

    def _read_split_items(self, mode):
        split_path = self._resolve_split_file(mode)
        if split_path is None:
            return []
        with open(split_path, 'r') as file:
            return [line.strip() for line in file.readlines() if line.strip()]

    def _expand_items_to_slices(self, items):
        expanded = []
        for item in items:
            sample_name = item[:-3] if item.endswith('.h5') else item
            if sample_name in self.all_slice_names:
                expanded.append(sample_name)
                continue
            patient_match = re.search(r'(patient\d+)', sample_name)
            if patient_match is not None:
                patient_id = patient_match.group(1)
                if patient_id in self.patient_to_slices:
                    expanded.extend(self.patient_to_slices[patient_id])
        return sorted(set(expanded))

    def _build_train_split(self):
        train_items = self._read_split_items('train')
        train_slice_names = self._expand_items_to_slices(train_items) if train_items else list(self.all_slice_names)
        patient_ids = sorted({self._extract_patient_id(name) for name in train_slice_names})
        rng = random.Random(1337)
        rng.shuffle(patient_ids)
        labeled_patient_num = max(0, min(int(self.args.labelnum), len(patient_ids)))
        labeled_patients = set(patient_ids[:labeled_patient_num])
        unlabeled_patients = set(patient_ids[labeled_patient_num:])
        labeled_names = [name for name in train_slice_names if self._extract_patient_id(name) in labeled_patients]
        unlabeled_names = [name for name in train_slice_names if self._extract_patient_id(name) in unlabeled_patients]
        if self.args.num is not None:
            unlabeled_names = unlabeled_names[:self.args.num]
        return labeled_names, unlabeled_names

    def _build_eval_list(self, mode):
        split_items = self._read_split_items(mode)
        if split_items:
            expanded = self._expand_items_to_slices(split_items)
            if expanded:
                return expanded
        return list(self.all_slice_names)

    def _sample_path(self, sample_name):
        return os.path.join(self.images_h5_dir, f'{sample_name}.h5')

    def _normalize_image(self, image):
        image = image.astype(np.float32)
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image, dtype=np.float32)
        return image

    def _resize_pair(self, image, label):
        target_h, target_w = self.crop_size
        image_pil = Image.fromarray(np.uint8(np.clip(image * 255.0, 0, 255)))
        label_pil = Image.fromarray(label.astype(np.uint8))
        image_pil = image_pil.resize((target_w, target_h), Image.BILINEAR)
        label_pil = label_pil.resize((target_w, target_h), Image.NEAREST)
        image = np.asarray(image_pil, dtype=np.float32) / 255.0
        label = np.asarray(label_pil, dtype=np.uint8)
        return image, label

    def _weak_augment(self, image, label):
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        elif random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        return image, label

    def _strong_augment(self, image):
        image_pil = Image.fromarray(np.uint8(np.clip(image * 255.0, 0, 255)))
        jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5)
        if random.random() < 0.8:
            image_pil = jitter(image_pil)
        if random.random() < 0.5:
            radius = np.random.uniform(0.1, 2.0)
            image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        image = np.asarray(image_pil, dtype=np.float32) / 255.0
        return image

    def _make_ovr_labels(self, label):
        ovr_labels = []
        for i in range(self.num_classes - 1):
            current = np.zeros_like(label, dtype=np.uint8)
            current[label == i + 1] = 1
            for pix in range(1, self.num_classes):
                if pix != i + 1:
                    current[label == pix] = 2
            ovr_labels.append(torch.from_numpy(current))
        return ovr_labels

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]
        with h5py.File(self._sample_path(sample_name), 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]
        image = self._normalize_image(image)
        if self.mode == 'train':
            image, label = self._resize_pair(image, label)
            image_weak, label_aug = self._weak_augment(image, label)
            image_strong = self._strong_augment(image_weak)
            sample = {
                'image': torch.from_numpy(image).float().unsqueeze(0),
                'image_weak': torch.from_numpy(image_weak.copy()).float().unsqueeze(0),
                'image_strong': torch.from_numpy(image_strong.copy()).float().unsqueeze(0),
                'label_aug': torch.from_numpy(label_aug.copy()).long(),
                'ovr_label': self._make_ovr_labels(label_aug),
                'idx': idx,
            }
            return sample
        return torch.from_numpy(image).float().unsqueeze(0), torch.from_numpy(label.astype(np.uint8)).long()

    def __len__(self):
        return len(self.sample_list)


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


import itertools


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
    if mask.ndim != 3:
        raise ValueError(f'OVR diagnosis mask should be [B, H, W], but got shape {tuple(mask.shape)}')
    mask_with_channel = torch.stack([mask] * num_classes, dim=1)
    return logits_map.detach(), mask.detach(), mask_with_channel.detach()


def build_dataloaders(args, cfg):
    trainset = CGSACDCDataset('train', args, cfg['crop_size'], cfg['nclass'])
    valset = CGSACDCDataset('val', args, cfg['crop_size'], cfg['nclass'])
    labeled_bs = args.labeled_bs if args.labeled_bs is not None else cfg['batch_size']
    unlabeled_bs = args.unlabeled_bs if args.unlabeled_bs is not None else cfg['batch_size']
    total_batch_size = labeled_bs + unlabeled_bs
    labeled_idxs = list(range(0, trainset.labeled_count))
    unlabeled_idxs = list(range(trainset.labeled_count, len(trainset)))
    if len(labeled_idxs) < labeled_bs or len(unlabeled_idxs) < unlabeled_bs:
        raise ValueError('Labeled or unlabeled sample count is smaller than requested batch size')
    batch_sampler = TwoStreamBatchSampler(
        primary_indices=labeled_idxs,
        secondary_indices=unlabeled_idxs,
        batch_size=total_batch_size,
        secondary_batch_size=unlabeled_bs,
    )
    trainloader = DataLoader(trainset, batch_sampler=batch_sampler, num_workers=cfg.get('num_workers', 4), pin_memory=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    return trainset, trainloader, valloader, labeled_bs


def build_model(class_num):
    model = OVRUNet(in_chns=1, class_num=class_num).cuda()
    return init_2d_weight(model)


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

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr if args.base_lr is not None else cfg['lr'],
        momentum=0.9,
        weight_decay=0.0001,
    )

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

    _, trainloader, valloader, labeled_bs = build_dataloaders(args, cfg)
    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d' % total_iters)
    class_names = build_class_names(cfg, args)

    dice_loss = MaskedDiceLoss(cfg['nclass']).cuda()
    dice_loss_ovr = MaskedDiceLoss(3).cuda()

    best_performance = 0.0
    best_epoch = 0
    epoch = -1
    iter_num = 0
    start_time = time.time()

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

    log_interval = max(len(trainloader) // 4, 1)
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum:{args.labelnum}, '
            f'best mdice:{best_performance:.4f} @epoch:{best_epoch}'
        )
        total_loss = AverageMeter()
        total_sup = AverageMeter()
        total_unsup = AverageMeter()
        model.train()
        ema_model.train()
        total_loader = zip(trainloader, trainloader)

        for i_batch, (sampled_batch, sample_batch_mix) in enumerate(total_loader):
            weak_batch = sampled_batch['image_weak'].cuda(non_blocking=True)
            strong_batch = sampled_batch['image_strong'].cuda(non_blocking=True)
            label_batch = sampled_batch['label_aug'].cuda(non_blocking=True)
            ovr_label = [item.cuda(non_blocking=True) for item in sampled_batch['ovr_label']]

            weak_batch_mix = sample_batch_mix['image_weak'].cuda(non_blocking=True)
            strong_batch_mix = sample_batch_mix['image_strong'].cuda(non_blocking=True)

            img_mask = random_mask(weak_batch.shape[0], weak_batch.shape[2], weak_batch.shape[3], args.cut_p)
            img_mask_bool = img_mask.bool()
            img_mask_channel = img_mask.unsqueeze(1)

            with torch.no_grad():
                out_mix, out_mix_ovr = ema_model(weak_batch_mix)
                soft_mix = torch.softmax(out_mix, dim=1)
                pred_mix = torch.argmax(soft_mix[labeled_bs:].detach(), dim=1)
                mix_conf_mask = torch.max(soft_mix, dim=1)[0] > args.conf_thresh

                soft_mix_ovr = [torch.softmax(item, dim=1) for item in out_mix_ovr]
                pred_mix_ovr = [torch.argmax(item[labeled_bs:].detach(), dim=1) for item in soft_mix_ovr]
                mix_conf_mask_ovr = generate_pseudo_label_withmask(soft_mix_ovr, cfg['nclass'])[1]
                mix_pseudo_outputs_ovr = torch.argmax(
                    generate_pseudo_label_withmask(soft_mix_ovr, cfg['nclass'])[0][labeled_bs:].detach(), dim=1
                )

            strong_batch[labeled_bs:] = (
                img_mask_channel[labeled_bs:] * strong_batch[labeled_bs:]
                + (1 - img_mask_channel[labeled_bs:]) * strong_batch_mix[labeled_bs:]
            )

            outputs_weak, outputs_weak_ovr = model(weak_batch)
            outputs_strong, outputs_strong_ovr = model(strong_batch)
            consistency_progress = iter_num / max(total_iters, 1) * args.consistency_rampup
            consistency_weight = get_current_consistency_weight(consistency_progress, args)

            sup_loss = F.cross_entropy(outputs_weak[:labeled_bs], label_batch[:labeled_bs].long()) + dice_loss(
                outputs_weak[:labeled_bs], label_batch[:labeled_bs]
            )
            sup_loss_ovr = 0.0
            for cls_idx in range(cfg['nclass'] - 1):
                sup_loss_ovr += F.cross_entropy(outputs_weak_ovr[cls_idx][:labeled_bs], ovr_label[cls_idx][:labeled_bs].long())
                sup_loss_ovr += dice_loss_ovr(outputs_weak_ovr[cls_idx][:labeled_bs], ovr_label[cls_idx][:labeled_bs])
            sup_loss = sup_loss + sup_loss_ovr / (cfg['nclass'] - 1)

            with torch.no_grad():
                ema_output, ema_ovr_output = ema_model(weak_batch)
                ema_outputs_soft = torch.softmax(ema_output, dim=1)
                ema_ovr_output_soft = [torch.softmax(item, dim=1) for item in ema_ovr_output]

                conf_mask = torch.max(ema_outputs_soft, dim=1)[0] > args.conf_thresh
                conf_mask = torch.where(img_mask_bool, conf_mask, mix_conf_mask)
                mask_channel = conf_mask.unsqueeze(1).expand(-1, cfg['nclass'], -1, -1).float()

                pseudo_outputs = torch.argmax(ema_outputs_soft[labeled_bs:].detach(), dim=1)
                pseudo_outputs = torch.where(img_mask_bool[labeled_bs:], pseudo_outputs, pred_mix)
                pseudo_lab4ovr = generate_ovr_pseudo_label(pseudo_outputs, cfg['nclass'])

                pseudo_logits_ovr, pseudo_conf_mask_ovr, _ = generate_pseudo_label_withmask(ema_ovr_output_soft, cfg['nclass'])
                conf_mask_ovr = torch.where(
                    img_mask_bool,
                    pseudo_conf_mask_ovr,
                    mix_conf_mask_ovr,
                )
                mask_channel_ovr = conf_mask_ovr.unsqueeze(1).expand(-1, cfg['nclass'], -1, -1).float()

                pseudo_ovr_full = []
                for cls_idx in range(cfg['nclass'] - 1):
                    pred_ovr = torch.argmax(ema_ovr_output_soft[cls_idx][labeled_bs:].detach(), dim=1)
                    pseudo_ovr_full.append(torch.where(img_mask_bool[labeled_bs:], pred_ovr, pred_mix_ovr[cls_idx]))

                pseudo_outputs_ovr = torch.argmax(pseudo_logits_ovr[labeled_bs:].detach(), dim=1)
                pseudo_outputs_ovr = torch.where(img_mask_bool[labeled_bs:], pseudo_outputs_ovr, mix_pseudo_outputs_ovr)

                ensemble_mask = pseudo_outputs == pseudo_outputs_ovr
                ensemble_mask_channel = ensemble_mask.unsqueeze(1).expand(-1, cfg['nclass'], -1, -1).float()
                ensemble_mask_channel_ovr = ensemble_mask.unsqueeze(1).expand(-1, 3, -1, -1).float()
                ensemble_pseudo = pseudo_outputs * ensemble_mask.long()
                ensemble_pseudo4ovr = generate_ovr_pseudo_label(ensemble_pseudo, cfg['nclass'])
                mask_channel_bin = conf_mask.unsqueeze(1).expand(-1, 3, -1, -1).float()
                mask_channel_ovr_bin = conf_mask_ovr.unsqueeze(1).expand(-1, 3, -1, -1).float()

            unsup_loss = cross_entropy_masked(outputs_strong[labeled_bs:], pseudo_outputs.long(), conf_mask[labeled_bs:])
            unsup_loss = unsup_loss + dice_loss(outputs_strong[labeled_bs:], pseudo_outputs, mask=mask_channel[labeled_bs:])

            unsup_loss = unsup_loss + cross_entropy_masked(
                outputs_strong[labeled_bs:], pseudo_outputs_ovr.long(), conf_mask_ovr[labeled_bs:]
            )
            unsup_loss = unsup_loss + dice_loss(
                outputs_strong[labeled_bs:], pseudo_outputs_ovr, mask=mask_channel_ovr[labeled_bs:]
            )

            unsup_loss = unsup_loss + cross_entropy_masked(
                outputs_strong[labeled_bs:], ensemble_pseudo.long(), ensemble_mask
            )
            unsup_loss = unsup_loss + dice_loss(
                outputs_strong[labeled_bs:], ensemble_pseudo, mask=ensemble_mask_channel
            )

            unsup_loss_ovr = 0.0
            for cls_idx in range(cfg['nclass'] - 1):
                unsup_loss_ovr += cross_entropy_masked(
                    outputs_strong_ovr[cls_idx][labeled_bs:], pseudo_lab4ovr[cls_idx].long(), conf_mask[labeled_bs:]
                )
                unsup_loss_ovr += dice_loss_ovr(
                    outputs_strong_ovr[cls_idx][labeled_bs:], pseudo_lab4ovr[cls_idx], mask=mask_channel_bin[labeled_bs:]
                )

                unsup_loss_ovr += cross_entropy_masked(
                    outputs_strong_ovr[cls_idx][labeled_bs:], pseudo_ovr_full[cls_idx].long(), conf_mask_ovr[labeled_bs:]
                )
                unsup_loss_ovr += dice_loss_ovr(
                    outputs_strong_ovr[cls_idx][labeled_bs:], pseudo_ovr_full[cls_idx], mask=mask_channel_ovr_bin[labeled_bs:]
                )

                unsup_loss_ovr += cross_entropy_masked(
                    outputs_strong_ovr[cls_idx][labeled_bs:], ensemble_pseudo4ovr[cls_idx].long(), ensemble_mask
                )
                unsup_loss_ovr += dice_loss_ovr(
                    outputs_strong_ovr[cls_idx][labeled_bs:], ensemble_pseudo4ovr[cls_idx], mask=ensemble_mask_channel_ovr
                )

            unsup_loss = unsup_loss + unsup_loss_ovr / (cfg['nclass'] - 1)
            loss = sup_loss + consistency_weight * unsup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = (args.base_lr if args.base_lr is not None else cfg['lr']) * (1.0 - iter_num / max(total_iters, 1)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            total_loss.update(loss.item())
            total_sup.update(sup_loss.item())
            total_unsup.update(unsup_loss.item())

            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/loss_all', loss.item(), iter_num)
            writer.add_scalar('train/loss_sup', sup_loss.item(), iter_num)
            writer.add_scalar('train/loss_unsup', unsup_loss.item(), iter_num)

            if i_batch % log_interval == 0:
                logger.info(
                    f'iteration {iter_num}/{total_iters} : loss {total_loss.avg:.4f} '
                    f'sup {total_sup.avg:.4f} unsup {total_unsup.avg:.4f}'
                )

        performance = None
        if iter_num >= total_iters * args.val_start and epoch % args.val_interval == 0:
            ema_eval_model = SegmentationWrapper(ema_model)
            performance, dice_class = eval_2d(valloader, ema_eval_model, cfg, ifdist=False, val_mode='ema')
            performance = performance.item()
            writer.add_scalar('eval/ema_mDice', performance, epoch)
            for cls_idx, dice in enumerate(dice_class):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                writer.add_scalar(f'eval/{class_name}_ema_DICE', dice.item(), epoch)
                logger.info(f'*** Evaluation: Class [{cls_idx + 1} {class_name}] Dice ema: {dice.item():.3f}')
            logger.info(f'*** Evaluation: MeanDice ema: {performance:.3f}')
            if performance >= best_performance:
                best_performance = performance
                best_epoch = epoch
                torch.save(
                    {'model': model.state_dict(), 'ema_model': ema_model.state_dict()},
                    os.path.join(cp_path, f'ep{epoch}_m{performance:.3f}.pth'),
                )
                logger.info(f'*** best checkpoint: MeanDice ema: {performance:.3f}, exp: {args.exp.split("_")[-1]}')

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

        if cfg.get('early_stop_patience') is not None and epoch - best_epoch >= cfg['early_stop_patience']:
            logger.info('Early stop.')
            break

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
        'CGS/Ep{}bs{}_{}_seed{}_label{}/thresh{}_{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, args.conf_thresh, args.exp
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
        else:
            print(f"Warning: {item} not found.")

    main(args, cfg, save_path, cp_path)
