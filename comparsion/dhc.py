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
import h5py
import numpy as np
from copy import deepcopy
from torch import nn
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.efficient import Flare_fixmatch_Dataset_effi, TwoStreamBatchSampler, mix_collate_fn
from models.model import unet_3D_wtcls, kaiming_normal_init_weight
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import count_params, init_log, AverageMeter
from utils.val import evaluate_3d


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
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints')
    parser.add_argument('--deterministic', type=str, default=False)
    parser.add_argument('--cps_w', type=float, default=1.0)
    parser.add_argument('--consistency_rampup', type=float, default=None)
    parser.add_argument('--cps_rampup', action='store_true', default=True)
    parser.add_argument('--no_cps_rampup', action='store_false', dest='cps_rampup')
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--no_mixed_precision', action='store_false', dest='mixed_precision')
    parser.add_argument('--accumulate_iters', type=int, default=50)
    parser.add_argument('--distdw_momentum', type=float, default=0.99)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--val_start_ratio', type=float, default=0.7)
    return parser


def ema_update(cur, prev, momentum):
    return momentum * prev + (1.0 - momentum) * cur


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, args, epochs):
    if not args.cps_rampup:
        return args.cps_w
    rampup_length = args.consistency_rampup if args.consistency_rampup is not None else epochs
    return args.cps_w * sigmoid_rampup(epoch, rampup_length)


def weighted_dice_loss(logits, target, class_weight, eps=1e-8):
    probs = torch.softmax(logits, dim=1)
    target_onehot = F.one_hot(target.long(), num_classes=probs.shape[1]).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = (probs * target_onehot).sum(dim=dims)
    denom = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    normalized_weight = class_weight / (class_weight.sum() + eps)
    return ((1.0 - dice) * normalized_weight).sum()


def supervised_loss(logits, target, class_weight):
    loss_ce = F.cross_entropy(logits, target.long(), weight=class_weight)
    loss_dice = weighted_dice_loss(logits, target, class_weight)
    return loss_ce + loss_dice


class DistDW:
    def __init__(self, num_cls, momentum=0.99):
        self.num_cls = num_cls
        self.momentum = momentum
        self.weights = None

    def _cal_weights(self, num_each_class, device):
        num_each_class = torch.tensor(num_each_class, dtype=torch.float32, device=device)
        p = (num_each_class.max() + 1e-8) / (num_each_class + 1e-8)
        p_log = torch.log(p + 1e-8)
        weight = p_log / (p_log.max() + 1e-8)
        return weight

    def init_weights(self, labeled_dataset, device):
        num_each_class = np.zeros(self.num_cls, dtype=np.float64)
        for data_id in labeled_dataset.name_list:
            with h5py.File(os.path.join(labeled_dataset.dir, 'labeled_h5', f'{data_id}.h5'), 'r') as h5f:
                label = h5f['label'][:].reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        self.weights = self._cal_weights(num_each_class, device) * self.num_cls
        return self.weights

    def get_ema_weights(self, pseudo_label):
        pseudo_label = torch.argmax(pseudo_label.detach(), dim=1, keepdim=False).long()
        label_numpy = pseudo_label.cpu().numpy()
        num_each_class = np.zeros(self.num_cls, dtype=np.float64)
        for i in range(label_numpy.shape[0]):
            label = label_numpy[i].reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        cur_weights = self._cal_weights(num_each_class, pseudo_label.device) * self.num_cls
        self.weights = ema_update(cur_weights, self.weights, momentum=self.momentum)
        return self.weights


class DiffDW:
    def __init__(self, num_cls, device, accumulate_iters=50):
        self.num_cls = num_cls
        self.last_dice = torch.zeros(num_cls, dtype=torch.float32, device=device) + 1e-8
        self.cls_learn = torch.zeros(num_cls, dtype=torch.float32, device=device)
        self.cls_unlearn = torch.zeros(num_cls, dtype=torch.float32, device=device)
        self.dice_weight = torch.ones(num_cls, dtype=torch.float32, device=device)
        self.accumulate_iters = accumulate_iters
        self.weights = None

    def init_weights(self):
        self.weights = torch.ones(self.num_cls, dtype=torch.float32, device=self.last_dice.device) * self.num_cls
        return self.weights

    def _per_class_dice(self, pred, label):
        output = torch.argmax(pred, dim=1)
        x_onehot = F.one_hot(output, num_classes=self.num_cls).permute(0, 4, 1, 2, 3).float()
        y_onehot = F.one_hot(label.long(), num_classes=self.num_cls).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        intersection = (x_onehot * y_onehot).sum(dim=dims)
        denom = x_onehot.sum(dim=dims) + y_onehot.sum(dim=dims)
        cur_dice = (2.0 * intersection + 1e-8) / (denom + 1e-8)
        return cur_dice

    def cal_weights(self, pred, label):
        cur_dice = self._per_class_dice(pred, label)
        delta_dice = cur_dice - self.last_dice
        ratio = torch.clamp(cur_dice / (self.last_dice + 1e-8), min=1e-8)
        cur_cls_learn = torch.where(delta_dice > 0, delta_dice, torch.zeros_like(delta_dice)) * torch.log(ratio)
        cur_cls_unlearn = torch.where(delta_dice <= 0, delta_dice, torch.zeros_like(delta_dice)) * torch.log(ratio)
        self.last_dice = cur_dice
        momentum = (self.accumulate_iters - 1) / self.accumulate_iters
        self.cls_learn = ema_update(cur_cls_learn, self.cls_learn, momentum=momentum)
        self.cls_unlearn = ema_update(cur_cls_unlearn, self.cls_unlearn, momentum=momentum)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
        cur_diff = torch.pow(torch.clamp(cur_diff, min=1e-8), 1.0 / 5.0)
        self.dice_weight = ema_update(1.0 - cur_dice, self.dice_weight, momentum=momentum)
        weights = cur_diff * self.dice_weight
        weights = weights / (weights.max() + 1e-8)
        self.weights = weights * self.num_cls
        return self.weights


class EnsembleModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x):
        return (self.model_a(x) + self.model_b(x)) / 2.0


def build_dataloaders(args, cfg):
    trainset_u = Flare_fixmatch_Dataset_effi('train_u', args, cfg['crop_size'])
    trainset_l = Flare_fixmatch_Dataset_effi('train_l', args, cfg['crop_size'])
    valset = Flare_fixmatch_Dataset_effi('val', args, cfg['crop_size'])
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
        secondary_batch_size=labeled_bs
    )
    trainloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=6,
        pin_memory=True,
        collate_fn=mix_collate_fn
    )
    return trainset_l, trainloader, valloader


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model_a = unet_3D_wtcls(in_chns=1, class_num=cfg['nclass']).cuda()
    model_b = unet_3D_wtcls(in_chns=1, class_num=cfg['nclass']).cuda()
    model_a = kaiming_normal_init_weight(model_a).cuda()
    model_b = kaiming_normal_init_weight(model_b).cuda()
    optimizer_a = AdamW(params=model_a.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_b = AdamW(params=model_b.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    scaler = GradScaler(enabled=args.mixed_precision)

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params A: {:.3f}M'.format(count_params(model_a)))
    logger.info('Total params B: {:.3f}M'.format(count_params(model_b)))

    trainset_l, trainloader, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d' % total_iters)

    diffdw = DiffDW(cfg['nclass'], device='cuda', accumulate_iters=args.accumulate_iters)
    distdw = DistDW(cfg['nclass'], momentum=args.distdw_momentum)
    weight_a = diffdw.init_weights()
    weight_b = distdw.init_weights(trainset_l, device='cuda')

    pre_best_dice = 0.0
    best_epoch = 0
    epoch = -1
    iter_num = 0
    start_time = time.time()

    latest_path = os.path.join(cp_path, 'latest.pth')
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model_a.load_state_dict(checkpoint['model_a'])
        model_b.load_state_dict(checkpoint['model_b'])
        optimizer_a.load_state_dict(checkpoint['optimizer_a'])
        optimizer_b.load_state_dict(checkpoint['optimizer_b'])
        diffdw.last_dice = checkpoint['diffdw_last_dice']
        diffdw.cls_learn = checkpoint['diffdw_cls_learn']
        diffdw.cls_unlearn = checkpoint['diffdw_cls_unlearn']
        diffdw.dice_weight = checkpoint['diffdw_dice_weight']
        diffdw.weights = checkpoint['weight_a']
        distdw.weights = checkpoint['weight_b']
        epoch = checkpoint['epoch']
        pre_best_dice = checkpoint['previous_best']
        best_epoch = checkpoint['best_epoch']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum:{args.labelnum}, '
            f'best mdice:{pre_best_dice:.4f} @epoch:{best_epoch}'
        )
        total_loss = AverageMeter()
        total_sup = AverageMeter()
        total_cps = AverageMeter()
        cps_w = get_current_consistency_weight(epoch, args, cfg['epochs'])
        model_a.train()
        model_b.train()

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_w, _, _, _ = unlabeled_data
            img_w = img_w.cuda()
            image = torch.cat([img_x, img_w], dim=0)
            labeled_bs = img_x.shape[0]

            optimizer_a.zero_grad()
            optimizer_b.zero_grad()

            with autocast(enabled=args.mixed_precision):
                output_a = model_a(image)
                output_b = model_b(image)
                output_a_l, output_a_u = output_a[:labeled_bs], output_a[labeled_bs:]
                output_b_l, output_b_u = output_b[:labeled_bs], output_b[labeled_bs:]
                max_a = torch.argmax(output_a.detach(), dim=1).long()
                max_b = torch.argmax(output_b.detach(), dim=1).long()
                weight_a = diffdw.cal_weights(output_a_l.detach(), mask_x.detach())
                weight_b = distdw.get_ema_weights(output_b_u.detach())
                loss_sup = supervised_loss(output_a_l, mask_x, weight_a) + supervised_loss(output_b_l, mask_x, weight_b)
                loss_cps = F.cross_entropy(output_a, max_b, weight=weight_a) + F.cross_entropy(output_b, max_a, weight=weight_b)
                loss = loss_sup + cps_w * loss_cps

            scaler.scale(loss).backward()
            scaler.step(optimizer_a)
            scaler.step(optimizer_b)
            scaler.update()

            total_loss.update(loss.item())
            total_sup.update(loss_sup.item())
            total_cps.update(loss_cps.item())
            iter_num += 1
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer_a.param_groups[0]['lr'] = lr
            optimizer_b.param_groups[0]['lr'] = lr

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_sup', loss_sup.item(), iters)
            writer.add_scalar('train/loss_cps', loss_cps.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/cps_w', cps_w, iters)

            if (i % max(args.num // 9, 1) == 0):
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, loss: {total_loss.avg:.3f}, '
                    f'sup: {total_sup.avg:.3f}, cps: {total_cps.avg:.3f}'
                )

        writer.add_scalars(
            'class_weights/A',
            {str(idx): float(weight_a[idx].item()) for idx in range(cfg['nclass'])},
            epoch
        )
        writer.add_scalars(
            'class_weights/B',
            {str(idx): float(weight_b[idx].item()) for idx in range(cfg['nclass'])},
            epoch
        )

        is_best = False
        if iter_num >= total_iters * args.val_start_ratio and epoch % args.val_interval == 0:
            ensemble_model = EnsembleModel(model_a, model_b)
            mDice, dice_class = evaluate_3d(valloader, ensemble_model, cfg, ifdist=False, val_mode='dhc')
            for cls_idx, dice in enumerate(dice_class):
                class_name = CLASSES[cfg['dataset']][cls_idx]
                writer.add_scalar(f'eval/{class_name}_dice', dice.item(), epoch)
                logger.info(f'*** Evaluation: Class [{cls_idx + 1} {class_name}] Dice: {dice:.3f}')
            writer.add_scalar('eval/mDice', mDice.item(), epoch)
            logger.info(f'*** Evaluation MeanDice: {mDice:.3f}')
            is_best = mDice.item() >= pre_best_dice
            if is_best:
                pre_best_dice = mDice.item()
                best_epoch = epoch
                model_ckpt = {'model_a': model_a.state_dict(), 'model_b': model_b.state_dict()}
                torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_mdice{mDice:.3f}.pth'))
                logger.info(f'*** best checkpoint: MeanDice {mDice:.3f}, exp: {args.exp.split("_")[-1]}')

        checkpoint = {
            'model_a': model_a.state_dict(),
            'model_b': model_b.state_dict(),
            'optimizer_a': optimizer_a.state_dict(),
            'optimizer_b': optimizer_b.state_dict(),
            'epoch': epoch,
            'previous_best': pre_best_dice,
            'best_epoch': best_epoch,
            'iter_num': iter_num,
            'start_time': start_time,
            'weight_a': weight_a.detach(),
            'weight_b': weight_b.detach(),
            'diffdw_last_dice': diffdw.last_dice.detach(),
            'diffdw_cls_learn': diffdw.cls_learn.detach(),
            'diffdw_cls_unlearn': diffdw.cls_unlearn.detach(),
            'diffdw_dice_weight': diffdw.dice_weight.detach()
        }
        torch.save(checkpoint, latest_path)

        if epoch >= (cfg['epochs'] - 1):
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
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cp_path = os.path.join(
        args.checkpoint_path,
        'DHC/Ep{}bs{}_{}_seed{}_label{}/cps{}_{}_{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'],
            args.seed, args.labelnum, args.cps_w, cfg['conf_t hresh'], args.exp
        )
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

    main(args, cfg, save_path, cp_path)
