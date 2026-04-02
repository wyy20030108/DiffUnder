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
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.efficient import ACDCsemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet, kaiming_normal_init_weight
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
    parser.add_argument('--deterministic', type=bool, default=False)
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    parser.add_argument('--no_mixed_precision', action='store_false', dest='mixed_precision')
    parser.add_argument('--cps_loss', type=str, default='wce')
    parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
    parser.add_argument('--cps_w', type=float, default=1.0)
    parser.add_argument('--cps_rampup', action='store_true', default=True)
    parser.add_argument('--no_cps_rampup', action='store_false', dest='cps_rampup')
    parser.add_argument('--consistency_rampup', type=float, default=None)
    parser.add_argument('--accumulate_iters', type=int, default=50)
    parser.add_argument('--dist_momentum', type=float, default=0.99)
    parser.add_argument('--val_interval', type=int, default=2)
    parser.add_argument('--val_start', type=float, default=0.5)
    return parser


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, args, max_epoch):
    if not args.cps_rampup:
        return args.cps_w
    rampup_length = args.consistency_rampup if args.consistency_rampup is not None else max_epoch
    return args.cps_w * sigmoid_rampup(epoch, rampup_length)


def ema_tensor(current, history, momentum):
    return momentum * history + (1.0 - momentum) * current


def one_hot_2d(target, num_classes):
    return F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def soft_dice_loss(logits, target, class_weight=None, ignore_index=None, eps=1e-8):
    probs = torch.softmax(logits, dim=1)
    if ignore_index is None:
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        safe_target = target.long()
    else:
        valid_mask = target != ignore_index
        safe_target = target.clone().long()
        safe_target[~valid_mask] = 0
    target_onehot = one_hot_2d(safe_target, logits.shape[1])
    valid_mask = valid_mask.unsqueeze(1).float()
    probs = probs * valid_mask
    target_onehot = target_onehot * valid_mask
    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dim=dims)
    denominator = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    if class_weight is None:
        weight = torch.ones_like(dice)
    else:
        weight = class_weight.float()
    weight = weight / weight.sum().clamp_min(eps)
    return ((1.0 - dice) * weight).sum()


def compute_seg_loss(name, logits, target, class_weight=None, ignore_index=None):
    if name == 'ce':
        return F.cross_entropy(logits, target.long(), ignore_index=-100 if ignore_index is None else ignore_index)
    if name == 'wce':
        return F.cross_entropy(logits, target.long(), weight=class_weight, ignore_index=-100 if ignore_index is None else ignore_index)
    if name == 'ce+dice':
        return F.cross_entropy(logits, target.long(), ignore_index=-100 if ignore_index is None else ignore_index) + soft_dice_loss(logits, target, ignore_index=ignore_index)
    if name == 'wce+dice':
        return F.cross_entropy(logits, target.long(), weight=class_weight, ignore_index=-100 if ignore_index is None else ignore_index) + soft_dice_loss(logits, target, ignore_index=ignore_index)
    if name == 'w_ce+dice':
        return F.cross_entropy(logits, target.long(), weight=class_weight, ignore_index=-100 if ignore_index is None else ignore_index) + soft_dice_loss(logits, target, class_weight=class_weight, ignore_index=ignore_index)
    raise ValueError(name)


class DistDW:
    def __init__(self, num_cls, momentum=0.99):
        self.num_cls = num_cls
        self.momentum = momentum
        self.weights = None

    def _cal_weights(self, num_each_class, device):
        num_each_class = torch.as_tensor(num_each_class, dtype=torch.float32, device=device)
        p = (num_each_class.max() + 1e-8) / (num_each_class + 1e-8)
        p_log = torch.log(p)
        return p_log / p_log.max().clamp_min(1e-8)

    def init_weights(self, labeled_dataset, device):
        num_each_class = np.zeros(self.num_cls, dtype=np.float64)
        sample_names = getattr(labeled_dataset, 'name_list', [])
        for sample_name in sample_names:
            if hasattr(labeled_dataset, '_get_sample_path'):
                sample_path = labeled_dataset._get_sample_path(sample_name)
            elif hasattr(labeled_dataset, '_sample_path'):
                sample_path = labeled_dataset._sample_path(sample_name)
            else:
                sample_path = os.path.join(labeled_dataset.images_h5_dir, f'{sample_name}.h5')
            with h5py.File(sample_path, 'r') as h5f:
                label = h5f['label'][:].reshape(-1)
            hist, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += hist
        self.weights = self._cal_weights(num_each_class, device) * self.num_cls
        return self.weights

    def get_ema_weights(self, pseudo_logits):
        pseudo_label = torch.argmax(pseudo_logits.detach(), dim=1).long()
        label_numpy = pseudo_label.cpu().numpy()
        num_each_class = np.zeros(self.num_cls, dtype=np.float64)
        for i in range(label_numpy.shape[0]):
            hist, _ = np.histogram(label_numpy[i].reshape(-1), range(self.num_cls + 1))
            num_each_class += hist
        cur_weights = self._cal_weights(num_each_class, pseudo_logits.device) * self.num_cls
        if self.weights is None:
            self.weights = cur_weights
        else:
            self.weights = ema_tensor(cur_weights, self.weights, self.momentum)
        return self.weights


class DiffDW:
    def __init__(self, num_cls, device, accumulate_iters=50):
        self.num_cls = num_cls
        self.accumulate_iters = accumulate_iters
        self.last_dice = torch.zeros(num_cls, dtype=torch.float32, device=device) + 1e-8
        self.cls_learn = torch.zeros(num_cls, dtype=torch.float32, device=device)
        self.cls_unlearn = torch.zeros(num_cls, dtype=torch.float32, device=device)
        self.dice_weight = torch.ones(num_cls, dtype=torch.float32, device=device)

    def init_weights(self):
        self.weights = torch.ones(self.num_cls, dtype=torch.float32, device=self.last_dice.device) * self.num_cls
        return self.weights

    def _per_class_dice(self, pred, label):
        pred_label = torch.argmax(pred, dim=1)
        pred_onehot = one_hot_2d(pred_label, self.num_cls)
        label_onehot = one_hot_2d(label.long(), self.num_cls)
        dims = (0, 2, 3)
        intersection = (pred_onehot * label_onehot).sum(dim=dims)
        denominator = pred_onehot.sum(dim=dims) + label_onehot.sum(dim=dims)
        return (2.0 * intersection + 1e-8) / (denominator + 1e-8)

    def cal_weights(self, pred, label):
        cur_dice = self._per_class_dice(pred, label)
        delta_dice = cur_dice - self.last_dice
        ratio = torch.clamp(cur_dice / self.last_dice.clamp_min(1e-8), min=1e-8)
        cur_cls_learn = torch.where(delta_dice > 0, delta_dice, torch.zeros_like(delta_dice)) * torch.log(ratio)
        cur_cls_unlearn = torch.where(delta_dice <= 0, delta_dice, torch.zeros_like(delta_dice)) * torch.log(ratio)
        self.last_dice = cur_dice
        momentum = (self.accumulate_iters - 1) / self.accumulate_iters
        self.cls_learn = ema_tensor(cur_cls_learn, self.cls_learn, momentum)
        self.cls_unlearn = ema_tensor(cur_cls_unlearn, self.cls_unlearn, momentum)
        cur_diff = torch.pow(torch.clamp((self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8), min=1e-8), 1.0 / 5.0)
        self.dice_weight = ema_tensor(1.0 - cur_dice, self.dice_weight, momentum)
        weights = cur_diff * self.dice_weight
        weights = weights / weights.max().clamp_min(1e-8)
        self.weights = weights * self.num_cls
        return self.weights


class EnsembleModel(torch.nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x):
        return (self.model_a(x) + self.model_b(x)) / 2.0


def build_class_names(cfg, args):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(args.dataset) or CLASSES.get(args.dataset.upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def build_dataloaders(args, cfg):
    trainset_u = ACDCsemiDataset('train_u', args, cfg['crop_size'])
    trainset_l = ACDCsemiDataset('train_l', args, cfg['crop_size'])
    valset = ACDCsemiDataset('val', args, cfg['crop_size'])
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    labeled_bs = int(cfg['batch_size'] / 2)
    unlabeled_bs = int(cfg['batch_size'] / 2)
    total_batch_size = labeled_bs + unlabeled_bs
    train_dataset = ConcatDataset([trainset_l, trainset_u])
    labeled_idxs = list(range(0, len(trainset_l)))
    unlabeled_idxs = list(range(len(trainset_l), len(trainset_l) + len(trainset_u)))
    if len(labeled_idxs) == 0 or len(unlabeled_idxs) == 0:
        raise ValueError('Both labeled and unlabeled datasets must be non-empty for DHC training')
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

    amp_enabled = args.mixed_precision and torch.cuda.is_available()
    model_a = UNet(in_chns=1, class_num=cfg['nclass']).cuda()
    model_b = UNet(in_chns=1, class_num=cfg['nclass']).cuda()
    model_a = kaiming_normal_init_weight(model_a).cuda()
    model_b = kaiming_normal_init_weight(model_b).cuda()
    optimizer_a = AdamW(params=model_a.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_b = AdamW(params=model_b.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    scaler = GradScaler(enabled=amp_enabled)

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params A: {:.3f}M'.format(count_params(model_a)))
    logger.info('Total params B: {:.3f}M'.format(count_params(model_b)))

    labeled_dataset, trainloader, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d' % total_iters)
    class_names = build_class_names(cfg, args)

    diffdw = DiffDW(cfg['nclass'], device='cuda', accumulate_iters=args.accumulate_iters)
    distdw = DistDW(cfg['nclass'], momentum=args.dist_momentum)
    weight_a = diffdw.init_weights()
    weight_b = distdw.init_weights(labeled_dataset, device='cuda')

    best_eval = 0.0
    best_epoch = 0
    epoch = -1
    iter_num = 0
    start_time = time.time()
    latest_ckpt = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, weights_only=False)
        model_a.load_state_dict(checkpoint['model_a'])
        model_b.load_state_dict(checkpoint['model_b'])
        optimizer_a.load_state_dict(checkpoint['optimizer_a'])
        optimizer_b.load_state_dict(checkpoint['optimizer_b'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        epoch = checkpoint['epoch']
        best_eval = checkpoint['best_eval']
        best_epoch = checkpoint['best_epoch']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        diffdw.last_dice = checkpoint['diffdw_last_dice'].cuda()
        diffdw.cls_learn = checkpoint['diffdw_cls_learn'].cuda()
        diffdw.cls_unlearn = checkpoint['diffdw_cls_unlearn'].cuda()
        diffdw.dice_weight = checkpoint['diffdw_dice_weight'].cuda()
        weight_a = checkpoint['weight_a'].cuda()
        weight_b = checkpoint['weight_b'].cuda()
        distdw.weights = weight_b
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    log_interval = max(len(trainloader) // 4, 1)
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, '
            f'labelnum:{args.labelnum}, best mdice:{best_eval:.4f} @epoch:{best_epoch}'
        )
        total_loss = AverageMeter()
        total_loss_sup = AverageMeter()
        total_loss_cps = AverageMeter()
        cps_w = get_current_consistency_weight(epoch, args, cfg['epochs'])
        model_a.train()
        model_b.train()
        current_eval = None

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)
            img_w, _, ignore_mask, _ = unlabeled_data
            img_w = img_w.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)
            image = torch.cat([img_x, img_w], dim=0)
            labeled_bs = img_x.shape[0]

            optimizer_a.zero_grad()
            optimizer_b.zero_grad()

            with autocast(enabled=amp_enabled):
                output_a = model_a(image)
                output_b = model_b(image)
                output_a_l = output_a[:labeled_bs]
                output_b_l, output_b_u = output_b[:labeled_bs], output_b[labeled_bs:]

                max_a = torch.argmax(output_a.detach(), dim=1).long()
                max_b = torch.argmax(output_b.detach(), dim=1).long()
                pseudo_a = max_a.clone()
                pseudo_b = max_b.clone()
                pseudo_a[labeled_bs:][ignore_mask == 255] = 255
                pseudo_b[labeled_bs:][ignore_mask == 255] = 255

                weight_a = diffdw.cal_weights(output_a_l.detach(), mask_x.detach())
                weight_b = distdw.get_ema_weights(output_b_u.detach())
                loss_sup = compute_seg_loss(args.sup_loss, output_a_l, mask_x, class_weight=weight_a) + compute_seg_loss(
                    args.sup_loss, output_b_l, mask_x, class_weight=weight_b
                )
                loss_cps = compute_seg_loss(args.cps_loss, output_a, pseudo_b, class_weight=weight_a, ignore_index=255) + compute_seg_loss(
                    args.cps_loss, output_b, pseudo_a, class_weight=weight_b, ignore_index=255
                )
                loss = loss_sup + cps_w * loss_cps

            scaler.scale(loss).backward()
            scaler.step(optimizer_a)
            scaler.step(optimizer_b)
            scaler.update()

            iter_num += 1
            total_loss.update(loss.item())
            total_loss_sup.update(loss_sup.item())
            total_loss_cps.update(loss_cps.item())
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer_a.param_groups[0]['lr'] = lr
            optimizer_b.param_groups[0]['lr'] = lr

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_sup', loss_sup.item(), iters)
            writer.add_scalar('train/loss_cps', loss_cps.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/cps_w', cps_w, iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {total_loss.avg:.3f}, '
                    f'loss_sup: {total_loss_sup.avg:.3f}, loss_cps: {total_loss_cps.avg:.3f}'
                )

        writer.add_scalars('class_weights/A', {str(i): float(weight_a[i].item()) for i in range(cfg['nclass'])}, epoch)
        writer.add_scalars('class_weights/B', {str(i): float(weight_b[i].item()) for i in range(cfg['nclass'])}, epoch)
        logger.info(f'epoch {epoch} : loss : {total_loss.avg}')
        logger.info(f'     Class Weights A: {[round(v, 4) for v in weight_a.detach().cpu().tolist()]}, lr: {optimizer_a.param_groups[0]["lr"]}')
        logger.info(f'     Class Weights B: {[round(v, 4) for v in weight_b.detach().cpu().tolist()]}')

        if iter_num >= total_iters * args.val_start and epoch % args.val_interval == 0:
            ensemble_model = EnsembleModel(model_a, model_b)
            current_eval, dice_class = eval_2d(valloader, ensemble_model, cfg, ifdist=False, val_mode='dhc')
            current_eval = current_eval.item()
            writer.add_scalar('eval/mDice', current_eval, epoch)
            for cls_idx, dice in enumerate(dice_class):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                writer.add_scalar(f'eval/{class_name}_DICE', dice.item(), epoch)
                logger.info(f'*** Evaluation: Class [{cls_idx + 1} {class_name}] Dice ensemble: {dice.item():.3f}')
            logger.info(f'*** Evaluation: MeanDice ensemble: {current_eval:.3f}')
            if current_eval >= best_eval:
                best_eval = current_eval
                best_epoch = epoch
                torch.save(
                    {'model_a': model_a.state_dict(), 'model_b': model_b.state_dict()},
                    os.path.join(cp_path, f'ep{epoch}_m{current_eval:.3f}.pth')
                )
                logger.info(f'*** best checkpoint: MeanDice ensemble: {current_eval:.3f}, exp: {args.exp.split("_")[-1]}')

        checkpoint = {
            'model_a': model_a.state_dict(),
            'model_b': model_b.state_dict(),
            'optimizer_a': optimizer_a.state_dict(),
            'optimizer_b': optimizer_b.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_eval': best_eval,
            'best_epoch': best_epoch,
            'iter_num': iter_num,
            'start_time': start_time,
            'weight_a': weight_a.detach().cpu(),
            'weight_b': weight_b.detach().cpu(),
            'diffdw_last_dice': diffdw.last_dice.detach().cpu(),
            'diffdw_cls_learn': diffdw.cls_learn.detach().cpu(),
            'diffdw_cls_unlearn': diffdw.cls_unlearn.detach().cpu(),
            'diffdw_dice_weight': diffdw.dice_weight.detach().cpu()
        }
        torch.save(checkpoint, latest_ckpt)

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
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    cp_path = os.path.join(
        args.checkpoint_path,
        'DHC/Ep{}bs{}_{}_seed{}_label{}/w{}_{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, args.cps_w, args.exp
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
        else:
            print(f"Warning: {item} not found.")

    main(args, cfg, save_path, cp_path)
