import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')

import argparse
import gc
import logging
import pprint
import random
import shutil
import sys
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Datasets.efficient import BUSISemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import AverageMeter, DiceLoss, count_params, init_log
from utils.val import eval_2d


SCRIPT_DEFAULT_CFG = {
    'dataset': 'BUSI',
    'base_dir': '/data/lhy_data/BUSI/Dataset_BUSI_with_GT/without_normal',
    'labelnum': 64,
    'label_help': 'number of labeled BUSI images',
    'num': None,
    'nclass': 2,
    'lr': 0.01,
    'epochs': 200,
    'batch_size': 8,
    'crop_size': 256,
    'conf_thresh': 0.95,
    'unsup_loss_weight': 0.1,
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


def get_default_args(cli_dataset):
    if cli_dataset in DATASET_CONFIGS:
        defaults = dict(DATASET_CONFIGS[cli_dataset])
        defaults.setdefault('dataset', cli_dataset)
        return defaults
    return dict(SCRIPT_DEFAULT_CFG)


def get_parser(defaults):
    parser = argparse.ArgumentParser(description=defaults.get('dataset', 'BUSI'))
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--base_dir', type=str, default=None)
    parser.add_argument(
        '--labelnum',
        type=int,
        default=None,
        help=defaults.get('label_help', SCRIPT_DEFAULT_CFG['label_help']),
    )
    parser.add_argument('--num', default=None, type=int, help='unlabeled data number')
    parser.add_argument('--config', type=str, default=defaults.get('config'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--ema_decay', type=float, default=0.996)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--val_num_workers', type=int, default=1)
    parser.add_argument('--agscore_reduced_size', type=int, default=64)
    parser.add_argument('--agscore_subsample_size', type=int, default=8192)
    parser.add_argument('--agscore_gmm_grid_size', type=int, default=512)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--val_start', type=float, default=0.3)
    parser.add_argument('--unsup_loss_weight', type=float, default=None)
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
    if args.unsup_loss_weight is not None:
        cfg['unsup_loss_weight'] = float(args.unsup_loss_weight)
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


@torch.no_grad()
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data.copy_(buffer.data)


class AgScoreLabelFilter:
    def __init__(self, num_pos_agents=64, num_neg_agents=256, reduced_size=64, subsample_size=8192, gmm_grid_size=512):
        self.N = num_pos_agents
        self.M = num_neg_agents
        self.reduced_size = reduced_size
        self.subsample_size = subsample_size
        self.gmm_grid_size = gmm_grid_size

    def _reduce_inputs(self, features, confidences, pseudo_labels):
        _, _, h, w = features.shape
        if max(h, w) <= self.reduced_size:
            return features, confidences, pseudo_labels, None
        reduced_h = min(h, self.reduced_size)
        reduced_w = min(w, self.reduced_size)
        reduced_features = F.adaptive_avg_pool2d(features, output_size=(reduced_h, reduced_w))
        reduced_confidences = F.adaptive_avg_pool2d(confidences.unsqueeze(1), output_size=(reduced_h, reduced_w)).squeeze(1)
        reduced_pseudo = F.interpolate(
            pseudo_labels.unsqueeze(1).float(),
            size=(reduced_h, reduced_w),
            mode='nearest',
        ).squeeze(1).long()
        return reduced_features, reduced_confidences, reduced_pseudo, (h, w)

    def _solve_threshold(self, gmm, fit_scores, correct_idx):
        lower = float(np.min(fit_scores))
        upper = float(np.max(fit_scores))
        if not np.isfinite(lower) or not np.isfinite(upper):
            return None
        if abs(upper - lower) < 1e-8:
            return None
        grid = np.linspace(lower, upper, num=self.gmm_grid_size, dtype=np.float64).reshape(-1, 1)
        probas = gmm.predict_proba(grid)[:, correct_idx]
        valid_indices = np.where(probas > 0.5)[0]
        if valid_indices.size == 0:
            return None
        return float(grid[valid_indices[0], 0])

    def _orthogonal_selection(self, candidates, count):
        num_cands = candidates.size(0)
        if num_cands == 0:
            return None

        cands_norm = F.normalize(candidates, p=2, dim=1)
        start_idx = torch.randint(0, num_cands, (1,), device=candidates.device).item()
        selected_feats = [candidates[start_idx]]
        selected_norm = [cands_norm[start_idx]]

        avail_mask = torch.ones(num_cands, dtype=torch.bool, device=candidates.device)
        avail_mask[start_idx] = False

        for _ in range(count - 1):
            if not avail_mask.any():
                selected_feats.append(selected_feats[-1])
                continue

            curr_selected_norm = torch.stack(selected_norm)
            avail_indices = torch.nonzero(avail_mask, as_tuple=False).squeeze(1)
            avail_cands_norm = cands_norm[avail_indices]
            sim_matrix = torch.mm(avail_cands_norm, curr_selected_norm.t())
            max_sim, _ = torch.max(sim_matrix, dim=1)
            best_idx_relative = torch.argmin(max_sim)
            best_idx_absolute = avail_indices[best_idx_relative]

            selected_feats.append(candidates[best_idx_absolute])
            selected_norm.append(cands_norm[best_idx_absolute])
            avail_mask[best_idx_absolute] = False

        return torch.stack(selected_feats)

    def _build_agents(self, features, confidences, pseudo_labels):
        _, channels, _, _ = features.shape
        feats_flat = features.permute(0, 2, 3, 1).reshape(-1, channels)
        conf_flat = confidences.reshape(-1)
        pl_flat = pseudo_labels.reshape(-1)

        unique_classes = torch.unique(pl_flat)
        pos_candidates_dict = {}
        neg_candidates_list = []

        for cls_idx in unique_classes:
            cls_mask = pl_flat == cls_idx
            cls_confs = conf_flat[cls_mask]
            cls_feats = feats_flat[cls_mask]
            if cls_confs.numel() == 0:
                continue
            cls_count = max(1, int(cls_confs.size(0) * 0.01))
            cls_count = min(cls_count, cls_confs.size(0))
            sorted_indices = torch.argsort(cls_confs)
            neg_candidates_list.append(cls_feats[sorted_indices[:cls_count]])
            pos_candidates_dict[cls_idx.item()] = cls_feats[sorted_indices[-cls_count:]]

        if not pos_candidates_dict or not neg_candidates_list:
            return None, None

        neg_candidates = torch.cat(neg_candidates_list, dim=0)

        pos_agents = []
        num_classes = len(pos_candidates_dict)
        samples_per_class = self.N // num_classes
        remainder = self.N % num_classes
        for order, (_, candidates) in enumerate(pos_candidates_dict.items()):
            sample_num = samples_per_class + (1 if order < remainder else 0)
            if sample_num <= 0 or candidates.size(0) == 0:
                continue
            if candidates.size(0) >= sample_num:
                chosen_idx = torch.randperm(candidates.size(0), device=candidates.device)[:sample_num]
            else:
                chosen_idx = torch.randint(0, candidates.size(0), (sample_num,), device=candidates.device)
            pos_agents.append(candidates[chosen_idx])

        if not pos_agents:
            return None, None

        pos_agents = torch.cat(pos_agents, dim=0)
        neg_agents = self._orthogonal_selection(neg_candidates, self.M)
        if neg_agents is None or pos_agents.numel() == 0:
            return None, None
        return pos_agents, neg_agents

    def _calculate_agscore(self, features, pos_agents, neg_agents):
        batch_size, channels, height, width = features.shape
        feats_flat = features.permute(0, 2, 3, 1).reshape(-1, channels)
        feats_norm = F.normalize(feats_flat, p=2, dim=1)
        pos_norm = F.normalize(pos_agents, p=2, dim=1)
        neg_norm = F.normalize(neg_agents, p=2, dim=1)

        sum_exp_pos = torch.sum(torch.exp(torch.mm(feats_norm, pos_norm.t())), dim=1)
        sum_exp_neg = torch.sum(torch.exp(torch.mm(feats_norm, neg_norm.t())), dim=1)
        ag_score = sum_exp_pos / (sum_exp_pos + sum_exp_neg).clamp_min(1e-8)
        return ag_score.reshape(batch_size, height, width)

    def get_pseudo_label_mask(self, features, confidences, pseudo_labels, fallback_threshold=0.95):
        batch_size, height, width = pseudo_labels.shape
        reduced_features, reduced_confidences, reduced_pseudo_labels, original_size = self._reduce_inputs(
            features,
            confidences,
            pseudo_labels,
        )

        pos_agents, neg_agents = self._build_agents(reduced_features, reduced_confidences, reduced_pseudo_labels)
        if pos_agents is None or neg_agents is None:
            return confidences >= fallback_threshold

        ag_scores = self._calculate_agscore(reduced_features, pos_agents, neg_agents)
        scores_np = ag_scores.detach().cpu().numpy().reshape(-1, 1)
        if scores_np.shape[0] == 0 or not np.isfinite(scores_np).all():
            return confidences >= fallback_threshold

        if scores_np.shape[0] > self.subsample_size:
            sampled_indices = np.random.choice(scores_np.shape[0], self.subsample_size, replace=False)
            fit_scores = scores_np[sampled_indices]
        else:
            fit_scores = scores_np
        fit_scores = fit_scores.astype(np.float64, copy=False)

        if fit_scores.shape[0] < 2:
            return confidences >= fallback_threshold
        if np.unique(np.round(fit_scores[:, 0], decimals=6)).shape[0] < 2:
            return confidences >= fallback_threshold
        if float(np.std(fit_scores[:, 0])) < 1e-6:
            return confidences >= fallback_threshold

        try:
            gmm = GaussianMixture(
                n_components=2,
                random_state=42,
                max_iter=100,
                n_init=5,
                tol=1e-3,
                reg_covar=1e-6,
                init_params='kmeans',
            )
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter('always', ConvergenceWarning)
                gmm.fit(fit_scores)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught_warnings):
                return confidences >= fallback_threshold
            if hasattr(gmm, 'converged_') and not gmm.converged_:
                return confidences >= fallback_threshold

            correct_idx = int(np.argmax(gmm.means_))
            threshold = self._solve_threshold(gmm, fit_scores, correct_idx)
            if threshold is None:
                return confidences >= fallback_threshold

            valid_mask = ag_scores >= threshold
            if original_size is not None:
                valid_mask = F.interpolate(
                    valid_mask.unsqueeze(1).float(),
                    size=original_size,
                    mode='nearest',
                ).squeeze(1) > 0.5
            return valid_mask.reshape(batch_size, height, width)
        except Exception:
            return confidences >= fallback_threshold


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model = init_2d_weight(UNet(in_chns=3, class_num=cfg['nclass']).cuda()).cuda()
    model_ema = deepcopy(model).cuda()
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    optimizer = AdamW(params=model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()

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
    unsup_loss_weight = float(cfg.get('unsup_loss_weight', 0.1))

    pre_best_dice = 0.0
    pre_best_dice_ema = 0.0
    best_epoch = 0
    best_epoch_ema = 0
    epoch = -1
    iter_num = 0

    latest_path = os.path.join(cp_path, 'latest.pth')
    best_model_path = os.path.join(cp_path, 'best_model.pth')
    best_ema_path = os.path.join(cp_path, 'best_ema.pth')
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
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

    ag_filter = AgScoreLabelFilter(
        num_pos_agents=64,
        num_neg_agents=256,
        reduced_size=args.agscore_reduced_size,
        subsample_size=args.agscore_subsample_size,
        gmm_grid_size=args.agscore_gmm_grid_size,
    )

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum:{args.labelnum}, '
            f'Previous best mdice model: {pre_best_dice:.4f} @epoch: {best_epoch}, '
            f'ema: {pre_best_dice_ema:.4f} @epoch: {best_epoch_ema}'
        )
        total_loss = AverageMeter()
        total_loss_l = AverageMeter()
        total_loss_u = AverageMeter()
        total_mask_ratio = AverageMeter()
        is_best = False
        model.train()

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)

            img_w, img_s, ignore_mask, _ = unlabeled_data
            img_w = img_w.cuda(non_blocking=True)
            img_s = img_s.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)

            pred_l = model(img_x)
            loss_ce = criterion_l(pred_l, mask_x)
            loss_dice = diceloss(pred_l, mask_x)
            loss_l = (loss_ce + loss_dice) / 2.0

            with torch.no_grad():
                pred_u_w, feature_u_w = model_ema(img_w, use_feature=True)
                pred_u_w = pred_u_w.detach()
                feature_u_w = feature_u_w.detach()
                pred_u_w_soft = torch.softmax(pred_u_w, dim=1)
                conf_u_w, mask_u_w = pred_u_w_soft.max(dim=1)
                ag_valid_mask = ag_filter.get_pseudo_label_mask(
                    feature_u_w,
                    conf_u_w,
                    mask_u_w,
                    fallback_threshold=cfg['conf_thresh'],
                )
                selected_mask_u = ag_valid_mask & (ignore_mask != 255)

            pred_u_student_s = model(img_s)
            loss_u_map = criterion_u(pred_u_student_s, mask_u_w)
            selected_mask_u_float = selected_mask_u.float()
            valid_pixel_count = selected_mask_u.sum().item()
            if valid_pixel_count > 0:
                loss_u = (loss_u_map * selected_mask_u_float).sum() / valid_pixel_count
                valid_region = (ignore_mask != 255).sum().item()
                mask_ratio = valid_pixel_count / max(valid_region, 1)
            else:
                loss_u = loss_u_map.sum() * 0.0
                mask_ratio = 0.0

            loss = loss_l + unsup_loss_weight * loss_u

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema_variables(model, model_ema, args.ema_decay, iter_num)

            total_loss.update(loss.item())
            total_loss_l.update(loss_l.item())
            total_loss_u.update(loss_u.item())
            total_mask_ratio.update(mask_ratio)
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            iter_num += 1

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/loss_l', loss_l.item(), iters)
            writer.add_scalar('train/loss_u', loss_u.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {total_loss.avg:.3f}, '
                    f'loss_l: {total_loss_l.avg:.3f}, loss_u: {total_loss_u.avg:.3f}, '
                    f'mask ratio: {total_mask_ratio.avg:.4f}'
                )

        if iter_num >= total_iters * args.val_start and epoch % args.val_interval == 0:
            model.eval()
            mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode='model')
            model.train()
            model_ema.eval()
            mDice_ema, dice_class_ema = eval_2d(valloader, model_ema, cfg, ifdist=False, val_mode='ema')
            model_ema.train()

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
                writer.add_scalar(f'eval/{class_name}_model_DICE', dice, epoch)
                writer.add_scalar(f'eval/{class_name}_ema_DICE', dice_class_ema[cls_idx], epoch)

            logger.info('*** Evaluation: MeanDice model: {:.3f}, ema: {:.3f}'.format(mDice, mDice_ema))
            writer.add_scalar('eval/mDice_model', mDice.item(), epoch)
            writer.add_scalar('eval/mDice_ema', mDice_ema.item(), epoch)

            model_improved = mDice.item() >= pre_best_dice
            ema_improved = mDice_ema.item() >= pre_best_dice_ema
            is_best = model_improved or ema_improved
            pre_best_dice = max(mDice.item(), pre_best_dice)
            pre_best_dice_ema = max(mDice_ema.item(), pre_best_dice_ema)
            if model_improved:
                best_epoch = epoch
                torch.save({'model': model.state_dict(), 'model_ema': model_ema.state_dict()}, best_model_path)
            if ema_improved:
                best_epoch_ema = epoch
                torch.save({'model': model.state_dict(), 'model_ema': model_ema.state_dict()}, best_ema_path)

        checkpoint = {
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': pre_best_dice,
            'previous_best_ema': pre_best_dice_ema,
            'best_epoch': best_epoch,
            'best_epoch_ema': best_epoch_ema,
            'iter_num': iter_num,
            'start_time': start_time,
        }
        torch.save(checkpoint, latest_path)
        model_ckpt = {'model': model.state_dict(), 'model_ema': model_ema.state_dict()}
        if is_best:
            logger.info(
                '*** best checkpoint: MeanDice model: {:.3f}, ema: {:.3f}\n*** exp: {}'.format(
                    pre_best_dice,
                    pre_best_dice_ema,
                    args.exp,
                )
            )
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_m{pre_best_dice:.3f}_ema{pre_best_dice_ema:.3f}.pth'))

    end_time = time.time()
    logger.info('Training time: {:.2f}s'.format(end_time - start_time))
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
        'AgScore_BUSI/Ep{}bs{}_{}_seed{}_label{}/thresh{}_{}'.format(
            cfg['epochs'],
            cfg['batch_size'],
            cfg['dataset'],
            args.seed,
            args.labelnum,
            cfg['conf_thresh'],
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
