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
from copy import deepcopy
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
from torch import nn
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


SCRIPT_DEFAULT_CFG = {
    'dataset': 'BUSI',
    'label_help': 'number of labeled BUSI images',
    'batch_size': 8,
    'conf_thresh': 0.95,
}


class SoftMatchWeighting:
    def __init__(self, num_classes, momentum=0.999, n_sigma=2.0, per_class=True, use_dist_align=True):
        self.num_classes = num_classes
        self.momentum = momentum
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.use_dist_align = use_dist_align
        self.p_model = None
        self.mu_t = None
        self.var_t = None

    def _init_state(self, device):
        if self.p_model is None:
            self.p_model = torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
        if self.mu_t is None:
            if self.per_class:
                self.mu_t = torch.full((self.num_classes,), 1.0 / self.num_classes, device=device)
                self.var_t = torch.ones(self.num_classes, device=device)
            else:
                self.mu_t = torch.tensor(1.0 / self.num_classes, device=device)
                self.var_t = torch.tensor(1.0, device=device)

    def align(self, probs):
        self._init_state(probs.device)
        if not self.use_dist_align:
            return probs

        batch_dist = probs.mean(dim=(0, 2, 3))
        self.p_model = self.momentum * self.p_model + (1.0 - self.momentum) * batch_dist
        target_dist = torch.full_like(self.p_model, 1.0 / self.num_classes)
        aligned = probs * (target_dist / self.p_model.clamp_min(1e-6)).view(1, -1, 1, 1)
        aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return aligned

    def update(self, max_probs, pseudo_labels, valid_mask):
        self._init_state(max_probs.device)
        if valid_mask.sum() == 0:
            return

        valid_probs = max_probs[valid_mask]
        valid_labels = pseudo_labels[valid_mask]

        if self.per_class:
            for class_idx in range(self.num_classes):
                class_mask = valid_labels == class_idx
                if class_mask.sum() == 0:
                    continue
                class_probs = valid_probs[class_mask]
                class_mean = class_probs.mean()
                class_var = class_probs.var(unbiased=False)
                self.mu_t[class_idx] = self.momentum * self.mu_t[class_idx] + (1.0 - self.momentum) * class_mean
                self.var_t[class_idx] = self.momentum * self.var_t[class_idx] + (1.0 - self.momentum) * class_var
        else:
            batch_mean = valid_probs.mean()
            batch_var = valid_probs.var(unbiased=False)
            self.mu_t = self.momentum * self.mu_t + (1.0 - self.momentum) * batch_mean
            self.var_t = self.momentum * self.var_t + (1.0 - self.momentum) * batch_var

    def weights(self, max_probs, pseudo_labels):
        self._init_state(max_probs.device)
        if self.per_class:
            mu = self.mu_t[pseudo_labels]
            var = self.var_t[pseudo_labels]
        else:
            mu = self.mu_t.expand_as(max_probs)
            var = self.var_t.expand_as(max_probs)

        diff = (mu - max_probs).clamp_min(0.0)
        denom = (2.0 * var.clamp_min(1e-6)) / max(self.n_sigma ** 2, 1e-6)
        weights = torch.exp(-(diff ** 2) / denom.clamp_min(1e-6))
        weights = torch.where(max_probs >= mu, torch.ones_like(weights), weights)
        return weights.clamp_(0.0, 1.0)

    def state_dict(self):
        return {
            'p_model': self.p_model,
            'mu_t': self.mu_t,
            'var_t': self.var_t,
        }

    def load_state_dict(self, state_dict):
        self.p_model = state_dict.get('p_model')
        self.mu_t = state_dict.get('mu_t')
        self.var_t = state_dict.get('var_t')


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


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(buffer.data)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('yes', 'true', 't', 'y', '1')


def get_default_args(cli_dataset):
    if cli_dataset in DATASET_CONFIGS:
        defaults = dict(DATASET_CONFIGS[cli_dataset])
        defaults.setdefault('dataset', cli_dataset)
        return defaults
    return {}


def get_parser(defaults):
    parser = argparse.ArgumentParser(description=defaults.get('dataset', 'BUSI'))
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument('--labelnum', type=int, default=None, help=defaults.get('label_help', SCRIPT_DEFAULT_CFG['label_help']))
    parser.add_argument('--num', default=None, type=int, help='unlabeled data number')
    parser.add_argument('--config', type=str, default=defaults.get('config'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--val_num_workers', type=int, default=1)
    parser.add_argument('--softmatch_momentum', type=float, default=0.999)
    parser.add_argument('--softmatch_n_sigma', type=float, default=2.0)
    parser.add_argument('--softmatch_per_class', type=str2bool, default=True)
    parser.add_argument('--dist_align', type=str2bool, default=True)
    parser.add_argument('--ulb_loss_ratio', type=float, default=0.1)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--val_start', type=float, default=0.3)
    return parser


def load_cfg(args, defaults):
    cfg = dict(SCRIPT_DEFAULT_CFG)
    cfg.update(defaults)
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg.update(yaml.load(f, Loader=yaml.Loader))
    for key in ('dataset', 'base_dir', 'labelnum', 'num'):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg.setdefault('dataset', 'BUSI')
    return cfg


def sync_args_from_cfg(args, cfg):
    for key in ('dataset', 'base_dir', 'labelnum', 'num', 'crop_size'):
        setattr(args, key, cfg.get(key))
    return args


def build_class_names(cfg):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(str(cfg.get('dataset')).upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model = init_2d_weight(UNet(in_chns=3, class_num=cfg['nclass']).cuda()).cuda()
    ema_model = init_2d_weight(deepcopy(model).cuda()).cuda()
    for param in ema_model.parameters():
        param.detach_()

    optimizer = AdamW(params=model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    softmatch = SoftMatchWeighting(
        num_classes=cfg['nclass'],
        momentum=args.softmatch_momentum,
        n_sigma=args.softmatch_n_sigma,
        per_class=args.softmatch_per_class,
        use_dist_align=args.dist_align,
    )

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

    trainset_u = BUSISemiDataset('train_u', args, cfg['crop_size'])
    trainset_l = BUSISemiDataset('train_l', args, cfg['crop_size'])
    valset = BUSISemiDataset('val', args, cfg['crop_size'])
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=args.val_num_workers,
        drop_last=False,
    )

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
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=mix_collate_fn,
    )

    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d', total_iters)
    class_names = build_class_names(cfg)
    log_interval = max(len(trainloader) // 8, 1)
    pre_best_dice = 0.0
    pre_best_dice_ema = 0.0
    best_epoch = 0
    best_epoch_ema = 0
    epoch = -1
    iter_num = 0
    latest_path = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        softmatch_state = checkpoint.get('softmatch')
        if softmatch_state is not None:
            softmatch.load_state_dict(softmatch_state)
        epoch = checkpoint['epoch']
        pre_best_dice = checkpoint['previous_best']
        pre_best_dice_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        start_time = time.time()

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum: {args.labelnum}, Previous best mdice '
            f'model: {pre_best_dice:.4f} @epoch: {best_epoch}, ema: {pre_best_dice_ema:.4f} @epoch: {best_epoch_ema}'
        )
        total_loss = AverageMeter()
        total_loss_l = AverageMeter()
        total_loss_u = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_weight_mean = AverageMeter()
        is_best = False
        model.train()
        ema_model.train()

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_w, img_s, ignore_mask, _ = unlabeled_data
            img_w, img_s = img_w.cuda(), img_s.cuda()
            ignore_mask = ignore_mask.cuda()

            pred_l = model(img_x)
            loss_ce = criterion_l(pred_l, mask_x)
            loss_dice = diceloss(pred_l, mask_x)
            loss_l = (loss_ce + loss_dice) / 2.0

            with torch.no_grad():
                pred_u_teacher = ema_model(img_w).detach()
                probs_u_teacher = torch.softmax(pred_u_teacher, dim=1)
                probs_u_teacher = softmatch.align(probs_u_teacher)
                conf_u, pseudo_u = probs_u_teacher.max(dim=1)
                valid_mask = ignore_mask != 255
                softmatch.update(conf_u, pseudo_u, valid_mask)
                pixel_weights = softmatch.weights(conf_u, pseudo_u) * valid_mask.float()

            pred_u_student = model(img_s)
            loss_u_map = criterion_u(pred_u_student, pseudo_u)
            valid_pixel_count = valid_mask.sum().item()
            if valid_pixel_count > 0:
                loss_u = (loss_u_map * pixel_weights).sum() / valid_pixel_count
                effective_mask_ratio = ((pixel_weights > 0.5) & valid_mask).float().sum().item() / valid_pixel_count
                weight_mean = pixel_weights[valid_mask].mean().item()
            else:
                loss_u = loss_u_map.sum() * 0.0
                effective_mask_ratio = 0.0
                weight_mean = 0.0

            loss = loss_l + args.ulb_loss_ratio * loss_u

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            total_loss.update(loss.item())
            total_loss_l.update(loss_l.item())
            total_loss_u.update(loss_u.item())
            total_mask_ratio.update(effective_mask_ratio)
            total_weight_mean.update(weight_mean)
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            iter_num += 1

            pseudo_fg_ratio = (pseudo_u > 0).float().mean().item()
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/loss_l', loss_l.item(), iters)
            writer.add_scalar('train/loss_u', loss_u.item(), iters)
            writer.add_scalar('train/mask_ratio', effective_mask_ratio, iters)
            writer.add_scalar('train/soft_weight_mean', weight_mean, iters)
            writer.add_scalar('train/pseudo_fg_ratio', pseudo_fg_ratio, iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {total_loss.avg:.3f}'
                    f', loss_l: {total_loss_l.avg:.3f}, loss_u: {total_loss_u.avg:.3f}'
                    f', soft ratio: {total_mask_ratio.avg:.4f}, soft weight: {total_weight_mean.avg:.4f}'
                    f', pseudo_fg: {pseudo_fg_ratio:.4f}'
                )

        if iter_num >= total_iters * args.val_start and epoch % args.val_interval == 0:
            model.eval()
            mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode='model')
            model.train()
            ema_model.eval()
            mDice_ema, dice_class_ema = eval_2d(valloader, ema_model, cfg, ifdist=False, val_mode='ema')
            ema_model.train()

            for cls_idx, dice in enumerate(dice_class):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info('*** Evaluation: Class [{:} {:}] Dice model: {:.3f}, ema: {:.3f}'.format(
                    cls_idx + 1, class_name, dice, dice_class_ema[cls_idx]
                ))
                writer.add_scalar(f'eval/{class_name}_model_DICE', dice, epoch)
                writer.add_scalar(f'eval/{class_name}_ema_DICE', dice_class_ema[cls_idx], epoch)

            logger.info('*** Evaluation: MeanDice model: {:.3f}, ema: {:.3f}'.format(mDice, mDice_ema))
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
            'optimizer': optimizer.state_dict(),
            'softmatch': softmatch.state_dict(),
            'epoch': epoch,
            'previous_best': pre_best_dice,
            'previous_best_ema': pre_best_dice_ema,
            'best_epoch': best_epoch,
            'best_epoch_ema': best_epoch_ema,
            'iter_num': iter_num,
            'start_time': start_time,
        }
        torch.save(checkpoint, latest_path)
        model_ckpt = {'model': model.state_dict(), 'ema_model': ema_model.state_dict()}
        if is_best:
            logger.info('*** best checkpoint: MeanDice model: {:.3f}, ema: {:.3f}\n*** exp: {}'.format(mDice, mDice_ema, args.exp))
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_m_{mDice:.3f}_ema_{mDice_ema:.3f}.pth'))

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
    args = sync_args_from_cfg(args, cfg)

    cp_path = os.path.join(
        args.checkpoint_path,
        'SoftMatch_BUSI/Ep{}bs{}_{}_seed{}_label{}/sigma{}_{}_{}'.format(
            cfg['epochs'],
            cfg['batch_size'],
            cfg['dataset'],
            args.seed,
            args.labelnum,
            args.softmatch_n_sigma,
            'da' if args.dist_align else 'noda',
            args.exp,
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
