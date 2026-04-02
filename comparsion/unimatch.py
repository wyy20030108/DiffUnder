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
import math
import re
import h5py
import numpy as np
from copy import deepcopy
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.unet2d import UNet
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
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--no_mixed_precision', action='store_false', dest='mixed_precision')
    parser.add_argument('--conf_thresh', type=float, default=0.95)
    parser.add_argument('--cutmix_p', type=float, default=0.5)
    parser.add_argument('--strong_p', type=float, default=0.8)
    parser.add_argument('--fp_noise_std', type=float, default=0.1)
    parser.add_argument('--fp_noise_clip', type=float, default=0.2)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--val_start', type=float, default=0.5)
    parser.add_argument('--resize_ratio_min', type=float, default=0.5)
    parser.add_argument('--resize_ratio_max', type=float, default=2.0)
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


def resize_with_ratio(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))
    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)
    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def crop_pair(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return img.crop((x, y, x + size, y + size)), mask.crop((x, y, x + size, y + size))


def hflip_pair(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def blur_image(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1 / 0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask
    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break
    mask[y:y + cutmix_h, x:x + cutmix_w] = 1
    return mask


def to_tensor_gray(img):
    img_arr = np.asarray(img, dtype=np.float32)
    if img_arr.max() > 1.0:
        img_arr = img_arr / 255.0
    return torch.from_numpy(img_arr).float().unsqueeze(0)


def normalize_h5_image(image):
    image = np.asarray(image, dtype=np.float32)
    image_min = image.min()
    image_max = image.max()
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image


class UniMatchACDCDataset(Dataset):
    def __init__(self, mode, args, size=None, nsample=None):
        self.mode = mode
        self.args = args
        self.size = self._normalize_size(size)
        self.root_path = args.base_dir
        self.images_h5_dir = self._resolve_images_h5_dir()
        self.all_slice_names = self._list_all_slice_names()
        self.patient_to_slices = self._build_patient_to_slices(self.all_slice_names)
        self.sample_names = self._build_sample_names()
        if self.mode == 'train_l' and nsample is not None and len(self.sample_names) > 0:
            repeat = math.ceil(nsample / len(self.sample_names))
            self.sample_names = (self.sample_names * repeat)[:nsample]
        print(f'{mode} data number is: {len(self.sample_names)}')

    def _normalize_size(self, size):
        if size is None:
            raise ValueError('size is required')
        if isinstance(size, int):
            return size
        if isinstance(size, (list, tuple)) and len(size) > 0:
            if len(size) == 1:
                return int(size[0])
            if int(size[-2]) != int(size[-1]):
                raise ValueError(f'UniMatchACDCDataset expects square crop size, but got {size}')
            return int(size[-1])
        raise ValueError('Invalid crop size')

    def _resolve_images_h5_dir(self):
        for folder in ['Images_h5', 'images_h5', 'data/slices', 'data']:
            folder_path = os.path.join(self.root_path, folder)
            if os.path.isdir(folder_path):
                return folder_path
        raise FileNotFoundError(f'Cannot find image h5 directory under {self.root_path}')

    def _list_all_slice_names(self):
        return sorted([name[:-3] for name in os.listdir(self.images_h5_dir) if name.endswith('.h5')])

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
            'train': ['train.txt', 'train.list', 'train_slice.txt', 'train_slices.list'],
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

    def _build_sample_names(self):
        if self.mode in ['train_l', 'train_u']:
            labeled_names, unlabeled_names = self._build_train_split()
            return labeled_names if self.mode == 'train_l' else unlabeled_names
        eval_mode = 'val' if self.mode == 'val' else 'test'
        split_items = self._read_split_items(eval_mode)
        if split_items:
            expanded = self._expand_items_to_slices(split_items)
            if expanded:
                return expanded
        return list(self.all_slice_names)

    def _sample_path(self, sample_name):
        return os.path.join(self.images_h5_dir, f'{sample_name}.h5')

    def __getitem__(self, index):
        sample_name = self.sample_names[index]
        with h5py.File(self._sample_path(sample_name), 'r') as h5f:
            image = normalize_h5_image(h5f['image'][:])
            if self.mode == 'train_u':
                label = np.zeros_like(image, dtype=np.uint8)
            else:
                label = h5f['label'][:].astype(np.uint8)

        image_pil = Image.fromarray(np.uint8(np.clip(image * 255.0, 0, 255)))
        mask_pil = Image.fromarray(label)

        if self.mode == 'val':
            return to_tensor_gray(image_pil), torch.from_numpy(label).long()

        ratio_range = (self.args.resize_ratio_min, self.args.resize_ratio_max)
        image_pil, mask_pil = resize_with_ratio(image_pil, mask_pil, ratio_range)
        ignore_value = 254 if self.mode == 'train_u' else 255
        image_pil, mask_pil = crop_pair(image_pil, mask_pil, self.size, ignore_value)
        image_pil, mask_pil = hflip_pair(image_pil, mask_pil, p=0.5)

        if self.mode == 'train_l':
            return to_tensor_gray(image_pil), torch.from_numpy(np.array(mask_pil)).long()

        img_w = deepcopy(image_pil)
        img_s1 = deepcopy(image_pil)
        img_s2 = deepcopy(image_pil)

        if random.random() < self.args.strong_p:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur_image(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=self.args.cutmix_p)

        if random.random() < self.args.strong_p:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur_image(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=self.args.cutmix_p)

        ignore_mask = torch.zeros((self.size, self.size), dtype=torch.long)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()
        ignore_mask[mask_tensor == 254] = 255

        return to_tensor_gray(img_w), to_tensor_gray(img_s1), to_tensor_gray(img_s2), ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.sample_names)


class FeaturePerturbationUNet(nn.Module):
    def __init__(self, in_chns, class_num, fp_noise_std=0.1, fp_noise_clip=0.2):
        super().__init__()
        self.net = init_2d_weight(UNet(in_chns=in_chns, class_num=class_num))
        self.fp_noise_std = fp_noise_std
        self.fp_noise_clip = fp_noise_clip

    def forward(self, x, return_fp=False):
        logits = self.net(x)
        if not return_fp:
            return logits
        noise = torch.clamp(torch.randn_like(x) * self.fp_noise_std, -self.fp_noise_clip, self.fp_noise_clip)
        logits_fp = self.net(x + noise)
        return logits, logits_fp


def build_dataloaders(args, cfg):
    trainset_u = UniMatchACDCDataset('train_u', args, cfg['crop_size'])
    trainset_l = UniMatchACDCDataset('train_l', args, cfg['crop_size'], nsample=len(trainset_u))
    valset = UniMatchACDCDataset('val', args, cfg['crop_size'])

    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.get('num_workers', 4),
        drop_last=True,
    )
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.get('num_workers', 4),
        drop_last=True,
    )
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    return trainloader_l, trainloader_u, valloader


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    amp_enabled = args.mixed_precision and torch.cuda.is_available()
    model = FeaturePerturbationUNet(
        in_chns=1,
        class_num=cfg['nclass'],
        fp_noise_std=args.fp_noise_std,
        fp_noise_clip=args.fp_noise_clip,
    ).cuda()
    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    scaler = GradScaler(device='cuda', enabled=amp_enabled)
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

    trainloader_l, trainloader_u, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg['epochs']
    logger.info('Total iters: %d' % total_iters)
    class_names = build_class_names(cfg, args)

    previous_best = 0.0
    best_epoch = 0
    epoch = -1
    iter_num = 0
    start_time = time.time()
    latest_path = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        best_epoch = checkpoint.get('best_epoch', 0)
        iter_num = checkpoint.get('iter_num', 0)
        start_time = checkpoint.get('start_time', start_time)
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    log_interval = max(len(trainloader_u) // 8, 1)
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum:{args.labelnum}, '
            f'Previous best mdice: {previous_best:.4f} @epoch:{best_epoch}'
        )
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)
            img_u_w = img_u_w.cuda(non_blocking=True)
            img_u_s1 = img_u_s1.cuda(non_blocking=True)
            img_u_s2 = img_u_s2.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)
            cutmix_box1 = cutmix_box1.cuda(non_blocking=True)
            cutmix_box2 = cutmix_box2.cuda(non_blocking=True)
            img_u_w_mix = img_u_w_mix.cuda(non_blocking=True)
            img_u_s1_mix = img_u_s1_mix.cuda(non_blocking=True)
            img_u_s2_mix = img_u_s2_mix.cuda(non_blocking=True)
            ignore_mask_mix = ignore_mask_mix.cuda(non_blocking=True)

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            cutmix_box1_mask = cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1
            cutmix_box2_mask = cutmix_box2.unsqueeze(1).expand_as(img_u_s2) == 1
            img_u_s1[cutmix_box1_mask] = img_u_s1_mix[cutmix_box1_mask]
            img_u_s2[cutmix_box2_mask] = img_u_s2_mix[cutmix_box2_mask]

            model.train()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            with autocast(device_type='cuda', enabled=amp_enabled):
                preds, preds_fp = model(torch.cat((img_x, img_u_w)), return_fp=True)
                pred_x, pred_u_w = preds.split([num_lb, num_ulb])
                pred_u_w_fp = preds_fp[num_lb:]
                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

                pred_u_w = pred_u_w.detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

                mask_u_w_cutmixed1 = mask_u_w.clone()
                conf_u_w_cutmixed1 = conf_u_w.clone()
                ignore_mask_cutmixed1 = ignore_mask.clone()

                mask_u_w_cutmixed2 = mask_u_w.clone()
                conf_u_w_cutmixed2 = conf_u_w.clone()
                ignore_mask_cutmixed2 = ignore_mask.clone()

                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
                ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
                ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

                loss_x = criterion_l(pred_x, mask_x)

                valid_mask_s1 = (conf_u_w_cutmixed1 >= args.conf_thresh) & (ignore_mask_cutmixed1 != 255)
                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                loss_u_s1 = loss_u_s1 * valid_mask_s1
                denom_s1 = max((ignore_mask_cutmixed1 != 255).sum().item(), 1)
                loss_u_s1 = loss_u_s1.sum() / denom_s1

                valid_mask_s2 = (conf_u_w_cutmixed2 >= args.conf_thresh) & (ignore_mask_cutmixed2 != 255)
                loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                loss_u_s2 = loss_u_s2 * valid_mask_s2
                denom_s2 = max((ignore_mask_cutmixed2 != 255).sum().item(), 1)
                loss_u_s2 = loss_u_s2.sum() / denom_s2

                valid_mask_fp = (conf_u_w >= args.conf_thresh) & (ignore_mask != 255)
                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                loss_u_w_fp = loss_u_w_fp * valid_mask_fp
                denom_fp = max((ignore_mask != 255).sum().item(), 1)
                loss_u_w_fp = loss_u_w_fp.sum() / denom_fp

                loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())

            valid_pixels = max((ignore_mask != 255).sum().item(), 1)
            mask_ratio = ((conf_u_w >= args.conf_thresh) & (ignore_mask != 255)).sum().item() / valid_pixels
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            iter_num += 1

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
            writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            writer.add_scalar('train/lr', lr, iters)

            if i % log_interval == 0:
                logger.info(
                    'Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: {:.3f}'.format(
                        i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_loss_w_fp.avg, total_mask_ratio.avg
                    )
                )

        if iter_num >= total_iters * args.val_start and epoch % args.val_interval == 0:
            model.eval()
            mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode='unimatch')
            mDice = mDice.item()

            for cls_idx, dice in enumerate(dice_class):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info('*** Evaluation: Class [{:} {:}] Dice: {:.3f}'.format(cls_idx + 1, class_name, dice.item()))
                writer.add_scalar(f'eval/{class_name}_DICE', dice.item(), epoch)
            logger.info('*** Evaluation MeanDice: {:.3f}'.format(mDice))
            writer.add_scalar('eval/mDice', mDice, epoch)

            is_best = mDice > previous_best
            previous_best = max(mDice, previous_best)
            if is_best:
                best_epoch = epoch
        else:
            is_best = False

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
            'best_epoch': best_epoch,
            'iter_num': iter_num,
            'start_time': start_time,
        }
        torch.save(checkpoint, latest_path)
        if is_best:
            torch.save(checkpoint, os.path.join(cp_path, f'ep{epoch}_m{previous_best:.3f}.pth'))
            logger.info('*** best checkpoint: MeanDice {:.3f} @epoch {}'.format(previous_best, best_epoch))

    end_time = time.time()
    logger.info('Training time: {:.2f}s'.format(end_time - start_time))
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
        'UniMatch/Ep{}bs{}_{}_seed{}_label{}/thresh{}_{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, args.conf_thresh, args.exp
        ),
    )
    os.makedirs(cp_path, exist_ok=True)
    save_path = os.path.join(cp_path, 'log')
    os.makedirs(save_path, exist_ok=True)

    include_list = ['methods', 'utils', 'configs', 'datasets_ours']
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
            print(f'Warning: {item} not found.')

    main(args, cfg, save_path, cp_path)
