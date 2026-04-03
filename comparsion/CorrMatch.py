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
from einops import rearrange
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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
    parser.add_argument('--config', type=str, default=cfgs['config'], help='Path to config file. If None, auto-generated based on labelnum')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=str, default=False)
    parser.add_argument('--consistency', type=float, default=1.0)
    parser.add_argument('--normal', type=bool, help='celoss normal or something-aware')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'])
    parser.add_argument('--threshold_momentum', type=float, default=0.999)
    parser.add_argument('--thresh_init', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--val_start', type=float, default=0.3)
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


class ThreshController:
    def __init__(self, nclass, momentum, thresh_init=0.85):
        self.nclass = nclass
        self.momentum = momentum
        self.thresh_global = torch.tensor(float(thresh_init)).cuda()

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        mask_pred = torch.argmax(pred, dim=1)
        pred_softmax = pred.softmax(dim=1)
        pred_conf = pred_softmax.max(dim=1)[0]
        unique_cls = torch.unique(mask_pred)
        cls_num = len(unique_cls)
        new_global = 0.0
        for cls in unique_cls:
            cls_map = mask_pred == cls
            if ignore_mask is not None:
                cls_map = cls_map & (ignore_mask != 255)
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            new_global += pred_conf[cls_map].max()
        if update_g and cls_num > 0:
            new_global = new_global / cls_num
            self.thresh_global = self.momentum * self.thresh_global + (1 - self.momentum) * new_global

    def get_thresh_global(self):
        return self.thresh_global


def build_class_names(cfg, args):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(args.dataset) or CLASSES.get(args.dataset.upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def make_supervised_loss(criterion_ce, criterion_dice, logits, target):
    loss_ce = criterion_ce(logits, target)
    loss_dice = criterion_dice(logits, target)
    return (loss_ce + loss_dice) / 2.0, loss_ce, loss_dice


def build_optimizer(model, args, cfg):
    if args.optimizer == 'SGD':
        return SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001, nesterov=True)
    return AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)


def build_dataloaders(args, cfg):
    trainset_u = ACDCsemiDataset('train_u', args, cfg['crop_size'])
    trainset_l = ACDCsemiDataset('train_l', args, cfg['crop_size'])
    valset = ACDCsemiDataset('val', args, cfg['crop_size'])
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=cfg['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    return trainloader_l, trainloader_u, trainloader_u_mix, valloader


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model = UNet(in_chns=1, class_num=cfg['nclass']).cuda()
    model = init_2d_weight(model).cuda()
    optimizer = build_optimizer(model, args, cfg)
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    criterion_kl = nn.KLDivLoss(reduction='none').cuda()

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

    trainloader_l, trainloader_u, trainloader_u_mix, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader_u) * cfg['epochs']
    logger.info('Total iters: %d' % total_iters)
    class_names = build_class_names(cfg, args)

    thresh_init = args.thresh_init if args.thresh_init is not None else cfg.get('thresh_init', cfg.get('conf_thresh', 0.95))
    thresh_controller = ThreshController(nclass=cfg['nclass'], momentum=args.threshold_momentum, thresh_init=thresh_init)

    pre_best_dice = 0.0
    best_epoch = 0
    epoch = -1
    iter_num = 0
    start_time = time.time()
    latest_path = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        pre_best_dice = checkpoint['previous_best']
        best_epoch = checkpoint['best_epoch']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        thresh_controller.thresh_global = checkpoint.get('thresh_global', thresh_controller.thresh_global).cuda()
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    log_interval = max(len(trainloader_u) // 8, 1)
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum:{args.labelnum}, '
            f'Previous best mdice: {pre_best_dice:.4f} @epoch:{best_epoch}, thresh:{thresh_controller.get_thresh_global().item():.4f}'
        )
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_kl = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_loss_corr_ce = AverageMeter()
        total_loss_corr_u = AverageMeter()
        total_mask_ratio = AverageMeter()

        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        model.train()
        for i, ((img_x, mask_x), (img_u_w, img_u_s1, ignore_mask, cutmix_box1), (img_u_w_mix, img_u_s1_mix, ignore_mask_mix, _)) in enumerate(loader):
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)
            img_u_w = img_u_w.cuda(non_blocking=True)
            img_u_s1 = img_u_s1.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)
            cutmix_box1 = cutmix_box1.cuda(non_blocking=True)
            img_u_w_mix = img_u_w_mix.cuda(non_blocking=True)
            img_u_s1_mix = img_u_s1_mix.cuda(non_blocking=True)
            ignore_mask_mix = ignore_mask_mix.cuda(non_blocking=True)

            with torch.no_grad():
                model.eval()
                res_u_w_mix = model(img_u_w_mix, need_fp=False, use_corr=False)
                pred_u_w_mix = res_u_w_mix if isinstance(res_u_w_mix, torch.Tensor) else res_u_w_mix['out']
                pred_u_w_mix = pred_u_w_mix.detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                cutmix_box1_mask = cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1
                img_u_s1[cutmix_box1_mask] = img_u_s1_mix[cutmix_box1_mask]

            model.train()
            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            res_w = model(torch.cat((img_x, img_u_w)), need_fp=True, use_corr=True)
            preds = res_w['out']
            preds_fp = res_w['out_fp']
            preds_corr = res_w['corr_out']
            preds_corr_map = res_w['corr_map'].detach()
            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            pred_u_w_corr_map = preds_corr_map[num_lb:]
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            res_s = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s['out']
            pred_u_s1_corr = res_s['corr_out']

            pred_u_w = pred_u_w.detach()
            if args.temperature != 1:
                pred_u_w = pred_u_w / args.temperature
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1 = mask_u_w.clone()
            conf_u_w_cutmixed1 = conf_u_w.clone()
            ignore_mask_cutmixed1 = ignore_mask.clone()
            corr_map_u_w_cutmixed1 = pred_u_w_corr_map.clone()
            b_sample, c_sample, _, _ = corr_map_u_w_cutmixed1.shape

            cutmix_box1_map = cutmix_box1 == 1
            mask_u_w_cutmixed1[cutmix_box1_map] = mask_u_w_mix[cutmix_box1_map]
            conf_u_w_cutmixed1[cutmix_box1_map] = conf_u_w_mix[cutmix_box1_map]
            ignore_mask_cutmixed1[cutmix_box1_map] = ignore_mask_mix[cutmix_box1_map]
            cutmix_box1_sample = rearrange(cutmix_box1_map, 'n h w -> n 1 h w')
            ignore_mask_cutmixed1_sample = rearrange((ignore_mask_cutmixed1 != 255), 'n h w -> n 1 h w')
            corr_map_u_w_cutmixed1 = (corr_map_u_w_cutmixed1 * (~cutmix_box1_sample) * ignore_mask_cutmixed1_sample).bool()

            thresh_controller.thresh_update(pred_u_w.detach(), ignore_mask_cutmixed1, update_g=True)
            thresh_global = thresh_controller.get_thresh_global()

            conf_filter_u_w = (conf_u_w_cutmixed1 >= thresh_global) & (ignore_mask_cutmixed1 != 255)
            conf_filter_u_w_without_cutmix = conf_filter_u_w.clone()
            conf_filter_u_w_sample = rearrange(conf_filter_u_w_without_cutmix, 'n h w -> n 1 h w')
            segments = (corr_map_u_w_cutmixed1 * conf_filter_u_w_sample).bool()

            for img_idx in range(b_sample):
                for segment_idx in range(c_sample):
                    segment = segments[img_idx, segment_idx]
                    segment_ori = corr_map_u_w_cutmixed1[img_idx, segment_idx]
                    if torch.sum(segment_ori) == 0:
                        continue
                    high_conf_ratio = torch.sum(segment).float() / torch.sum(segment_ori).float()
                    if torch.sum(segment) == 0 or high_conf_ratio < thresh_global:
                        continue
                    unique_cls, count = torch.unique(mask_u_w_cutmixed1[img_idx][segment == 1], return_counts=True)
                    if len(unique_cls) == 0:
                        continue
                    if torch.max(count).float() / torch.sum(count).float() > thresh_global:
                        top_class = unique_cls[torch.argmax(count)]
                        mask_u_w_cutmixed1[img_idx][segment_ori == 1] = top_class
                        conf_filter_u_w_without_cutmix[img_idx] = conf_filter_u_w_without_cutmix[img_idx] | segment_ori
            conf_filter_u_w_without_cutmix = conf_filter_u_w_without_cutmix | conf_filter_u_w

            loss_x, loss_ce_x, loss_dice_x = make_supervised_loss(criterion_l, diceloss, pred_x, mask_x)
            loss_x_corr, _, _ = make_supervised_loss(criterion_l, diceloss, pred_x_corr, mask_x)

            valid_s1 = max(torch.sum(ignore_mask_cutmixed1 != 255).item(), 1)
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * conf_filter_u_w_without_cutmix.float()
            loss_u_s1 = torch.sum(loss_u_s1) / valid_s1

            loss_u_corr_s1 = criterion_u(pred_u_s1_corr, mask_u_w_cutmixed1)
            loss_u_corr_s1 = loss_u_corr_s1 * conf_filter_u_w_without_cutmix.float()
            loss_u_corr_s1 = torch.sum(loss_u_corr_s1) / valid_s1

            valid_w = max(torch.sum(ignore_mask != 255).item(), 1)
            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            loss_u_corr_w = loss_u_corr_w * (((conf_u_w >= thresh_global) & (ignore_mask != 255)).float())
            loss_u_corr_w = torch.sum(loss_u_corr_w) / valid_w
            loss_u_corr = 0.5 * (loss_u_corr_s1 + loss_u_corr_w)

            softmax_pred_u_w = F.softmax(pred_u_w.detach(), dim=1)
            logsoftmax_pred_u_s1 = F.log_softmax(pred_u_s1, dim=1)
            loss_u_kl = criterion_kl(logsoftmax_pred_u_s1, softmax_pred_u_w)
            loss_u_kl = torch.sum(loss_u_kl, dim=1) * conf_filter_u_w.float()
            loss_u_kl = torch.sum(loss_u_kl) / valid_s1

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * (((conf_u_w >= thresh_global) & (ignore_mask != 255)).float())
            loss_u_w_fp = torch.sum(loss_u_w_fp) / valid_w

            loss = (0.5 * loss_x + 0.5 * loss_x_corr + loss_u_s1 * 0.25 + loss_u_kl * 0.25 + loss_u_w_fp * 0.25 + 0.25 * loss_u_corr) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s1.item())
            total_loss_kl.update(loss_u_kl.item())
            total_loss_w_fp.update(loss_u_w_fp.item())
            total_loss_corr_ce.update(loss_x_corr.item())
            total_loss_corr_u.update(loss_u_corr.item())
            mask_ratio = ((conf_u_w >= thresh_global) & (ignore_mask != 255)).sum().item() / valid_w
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            iter_num += 1

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_ce_x', loss_ce_x.item(), iters)
            writer.add_scalar('train/loss_dice_x', loss_dice_x.item(), iters)
            writer.add_scalar('train/loss_corr_ce', loss_x_corr.item(), iters)
            writer.add_scalar('train/loss_s', loss_u_s1.item(), iters)
            writer.add_scalar('train/loss_kl', loss_u_kl.item(), iters)
            writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
            writer.add_scalar('train/loss_corr_u', loss_u_corr.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            writer.add_scalar('train/thresh_global', thresh_global.item(), iters)
            writer.add_scalar('train/lr', lr, iters)

            if i % log_interval == 0:
                logger.info(
                    'Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, loss_corr_ce: {:.3f}, '
                    'Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask: {:.3f}, loss_corr_u: {:.3f}'.format(
                        i,
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_corr_ce.avg,
                        total_loss_s.avg,
                        total_loss_w_fp.avg,
                        total_mask_ratio.avg,
                        total_loss_corr_u.avg,
                    )
                )

        is_best = False
        if iter_num >= total_iters * args.val_start and epoch % args.val_interval == 0:
            model.eval()
            mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode='CorrMatch')
            for cls_idx, dice in enumerate(dice_class):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info('*** Evaluation: Class [{:} {:}] Dice: {:.3f}'.format(cls_idx + 1, class_name, dice.item()))
                writer.add_scalar(f'eval/{class_name}_DICE', dice.item(), epoch)
            logger.info('*** Evaluation MeanDice: {:.3f}'.format(mDice.item()))
            writer.add_scalar('eval/mDice', mDice.item(), epoch)
            is_best = mDice.item() >= pre_best_dice
            pre_best_dice = max(mDice.item(), pre_best_dice)
            if is_best:
                best_epoch = epoch

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': pre_best_dice,
            'best_epoch': best_epoch,
            'iter_num': iter_num,
            'start_time': start_time,
            'thresh_global': thresh_controller.get_thresh_global().detach().cpu(),
        }
        torch.save(checkpoint, latest_path)
        if is_best:
            model_ckpt = {'model': model.state_dict()}
            best_path = os.path.join(cp_path, f'ep{epoch}_m{pre_best_dice:.3f}.pth')
            torch.save(model_ckpt, best_path)
            logger.info('save model to {}'.format(best_path))

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
    print(dataset_name)
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
        'CorrMatch/Ep{}bs{}_{}_seed{}_label{}/thresh{}_{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum,
            cfg.get('conf_thresh', args.thresh_init if args.thresh_init is not None else 0.95), args.exp
        )
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
