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
from torch import nn
from torch.optim import AdamW, Adam, SGD
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.efficient import BUSISemiDataset, TwoStreamBatchSampler, mix_collate_fn
from Datasets.transform import obtain_cutmix_box
from models.unet2d import UNet
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
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--conf_thresh', type=float, default=defaults.get('conf_thresh', DEFAULT_BUSI_CFG['conf_thresh']))
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--pert_gap', type=float, default=0.5)
    parser.add_argument('--pert_type', type=str, default='dropout')
    return parser


def load_cfg(args, defaults):
    cfg = dict(DEFAULT_BUSI_CFG)
    cfg.update(defaults)
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg.update(yaml.load(f, Loader=yaml.Loader))
    cfg['dataset'] = args.dataset
    cfg['conf_thresh'] = args.conf_thresh
    return cfg


def build_class_names(cfg):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(str(cfg.get('dataset')).upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def build_optimizer(model, args, cfg):
    optimizer_name = args.optimizer.lower()
    if optimizer_name == 'sgd':
        return SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001, nesterov=True)
    if optimizer_name == 'adam':
        return Adam(model.parameters(), lr=cfg['lr'], weight_decay=0.0001)
    if optimizer_name == 'adamw':
        return AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    raise NotImplementedError(args.optimizer)


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


def make_second_cutmix_box(batch_size, image_size, device):
    boxes = [obtain_cutmix_box(image_size).to(device) for _ in range(batch_size)]
    return torch.stack(boxes, dim=0)


class SoftDiceLossIgnore(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, inputs, target, ignore=None):
        if target.ndim == 3:
            target = target.unsqueeze(1)
        target = target.long()
        if target.size(1) == 1:
            target = target.squeeze(1)
            target_onehot = torch.nn.functional.one_hot(
                target.clamp(min=0, max=self.n_classes - 1),
                num_classes=self.n_classes,
            ).permute(0, 3, 1, 2).float()
        else:
            target_onehot = target.float()
        probs = inputs
        if ignore is None:
            valid_mask = torch.ones_like(target_onehot)
        else:
            if ignore.ndim == 3:
                ignore = ignore.unsqueeze(1)
            valid_mask = (1.0 - ignore.float()).expand_as(target_onehot)
        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2.0 * intersection + 1e-10) / (denominator + 1e-10)
        return 1.0 - dice.mean()


class PertDropout(nn.Module):
    def __init__(self, p=0.5, dropout_type='dropout'):
        super().__init__()
        top = min(max(0.5 + p / 2.0, 0.0), 1.0)
        bottom = min(max(0.5 - p / 2.0, 0.0), 1.0)
        dropout_cls = {
            'dropout': nn.Dropout2d,
            'alpha': nn.AlphaDropout,
            'feature': nn.FeatureAlphaDropout,
        }
        if dropout_type not in dropout_cls:
            raise ValueError(f'Unsupported pert_type: {dropout_type}')
        self.dropouts = nn.ModuleList([
            dropout_cls[dropout_type](bottom),
            dropout_cls[dropout_type](top),
        ])

    def __len__(self):
        return len(self.dropouts)

    def forward(self, features):
        results = []
        for dropout in self.dropouts:
            results.append([dropout(feat) for feat in features])
        return results


class CrossMatchUNet(nn.Module):
    def __init__(self, in_chns, class_num, pert_gap=0.5, pert_type='dropout'):
        super().__init__()
        self.net = init_2d_weight(UNet(in_chns=in_chns, class_num=class_num))
        self.pert = PertDropout(p=pert_gap, dropout_type=pert_type)

    def forward(self, x, need_fp=False):
        features = self.net.encoder(x)
        if need_fp:
            if x.shape[0] % 2 != 0:
                raise ValueError('need_fp=True expects an even batch size to split source and target halves')
            features_x = []
            features_u = []
            for feat in features:
                fx, fu = feat.chunk(2)
                features_x.append(fx)
                features_u.append(fu)
            perturbed_u = self.pert(features_u)
            decoder_inputs = [
                torch.cat(feature_group, dim=0)
                for feature_group in zip(features_x, features_u, *perturbed_u)
            ]
            decoder_feature = self.net.decoder(decoder_inputs)
            logits = self.net.classifier(decoder_feature)
            return logits.chunk(2 + len(self.pert), dim=0)
        decoder_feature = self.net.decoder(features)
        return self.net.classifier(decoder_feature)


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model = CrossMatchUNet(
        in_chns=3,
        class_num=cfg['nclass'],
        pert_gap=args.pert_gap,
        pert_type=args.pert_type,
    ).cuda()
    optimizer = build_optimizer(model, args, cfg)
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    criterion_dice = SoftDiceLossIgnore(cfg['nclass']).cuda()

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

    trainloader, valloader = build_dataloaders(args, cfg)
    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d', total_iters)
    class_names = build_class_names(cfg)
    log_interval = max(len(trainloader) // 8, 1)

    previous_best = 0.0
    best_epoch = 0
    epoch = -1
    iter_num = 0
    latest_path = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        best_epoch = checkpoint['best_epoch']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        start_time = time.time()

    conf_thresh = cfg['conf_thresh']
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum: {args.labelnum}, '
            f'Previous best mdice: {previous_best:.4f} @epoch: {best_epoch}'
        )
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_kd = AverageMeter()
        total_mask_ratio = AverageMeter()
        model.train()
        is_best = False

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_u_w, img_u_s, ignore_mask, cutmix_box1 = unlabeled_data
            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)
            img_u_w = img_u_w.cuda(non_blocking=True)
            img_u_s = img_u_s.cuda(non_blocking=True)
            ignore_mask = ignore_mask.cuda(non_blocking=True)
            cutmix_box1 = cutmix_box1.cuda(non_blocking=True)

            num_ulb = img_u_w.shape[0]
            mix_perm = torch.randperm(num_ulb, device=img_u_w.device)
            img_u_w_mix = img_u_w[mix_perm]
            img_u_s1_mix = img_u_s[mix_perm]
            img_u_s2_mix = img_u_s[mix_perm]
            ignore_mask_mix = ignore_mask[mix_perm]

            img_u_s1 = img_u_s.clone()
            img_u_s2 = img_u_s.clone()
            cutmix_box2 = make_second_cutmix_box(num_ulb, img_u_s.shape[-1], img_u_s.device)

            with torch.no_grad():
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                _, _, pred_u_w_weak_mix, pred_u_w_strong_mix = model(
                    torch.cat((img_x, img_u_w_mix), dim=0),
                    need_fp=True,
                )
                conf_u_w_weak_mix = pred_u_w_weak_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_weak_mix = pred_u_w_weak_mix.argmax(dim=1)
                conf_u_w_strong_mix = pred_u_w_strong_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_strong_mix = pred_u_w_strong_mix.argmax(dim=1)

            cutmix_box1_mask = cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1
            cutmix_box2_mask = cutmix_box2.unsqueeze(1).expand_as(img_u_s2) == 1
            img_u_s1[cutmix_box1_mask] = img_u_s1_mix[cutmix_box1_mask]
            img_u_s2[cutmix_box2_mask] = img_u_s2_mix[cutmix_box2_mask]

            pred_x, pred_u_w, pred_u_w_weak, pred_u_w_strong = model(
                torch.cat((img_x, img_u_w), dim=0),
                need_fp=True,
            )
            pred_u_s1, pred_u_s2, pred_u_s2_weak, pred_u_s2_strong = model(
                torch.cat((img_u_s1, img_u_s2), dim=0),
                need_fp=True,
            )

            if args.temperature != 1:
                pred_u_w = pred_u_w / args.temperature
                pred_u_w_weak = pred_u_w_weak / args.temperature
                pred_u_w_strong = pred_u_w_strong / args.temperature

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            conf_u_w_weak = pred_u_w_weak.softmax(dim=1).max(dim=1)[0]
            mask_u_w_weak = pred_u_w_weak.argmax(dim=1)

            conf_u_w_strong = pred_u_w_strong.softmax(dim=1).max(dim=1)[0]
            mask_u_w_strong = pred_u_w_strong.argmax(dim=1)

            mask_u_w_cutmixed1 = mask_u_w.clone()
            conf_u_w_cutmixed1 = conf_u_w.clone()
            ignore_mask_cutmixed1 = ignore_mask.clone()

            mask_u_w_cutmixed2 = mask_u_w.clone()
            conf_u_w_cutmixed2 = conf_u_w.clone()
            ignore_mask_cutmixed2 = ignore_mask.clone()

            mask_u_w_weak_cutmixed2 = mask_u_w_weak.clone()
            conf_u_w_weak_cutmixed2 = conf_u_w_weak.clone()

            mask_u_w_strong_cutmixed2 = mask_u_w_strong.clone()
            conf_u_w_strong_cutmixed2 = conf_u_w_strong.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            mask_u_w_weak_cutmixed2[cutmix_box2 == 1] = mask_u_w_weak_mix[cutmix_box2 == 1]
            conf_u_w_weak_cutmixed2[cutmix_box2 == 1] = conf_u_w_weak_mix[cutmix_box2 == 1]

            mask_u_w_strong_cutmixed2[cutmix_box2 == 1] = mask_u_w_strong_mix[cutmix_box2 == 1]
            conf_u_w_strong_cutmixed2[cutmix_box2 == 1] = conf_u_w_strong_mix[cutmix_box2 == 1]

            ignore_unsup_1 = ((conf_u_w_cutmixed1 < conf_thresh) | (ignore_mask_cutmixed1 == 255)).float()
            ignore_unsup_2 = ((conf_u_w_cutmixed2 < conf_thresh) | (ignore_mask_cutmixed2 == 255)).float()
            ignore_weak = ((conf_u_w_weak_cutmixed2 < conf_thresh) | (ignore_mask_cutmixed2 == 255)).float()
            ignore_strong = ((conf_u_w_strong_cutmixed2 < conf_thresh) | (ignore_mask_cutmixed2 == 255)).float()
            ignore_teacher = ((conf_u_w < conf_thresh) | (ignore_mask == 255)).float()

            loss_x = (
                criterion_l(pred_x, mask_x)
                + criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())
            ) / 2.0

            loss_u_s1 = criterion_dice(
                pred_u_s1.softmax(dim=1),
                mask_u_w_cutmixed1.unsqueeze(1).float(),
                ignore=ignore_unsup_1,
            )
            loss_u_s2 = criterion_dice(
                pred_u_s2.softmax(dim=1),
                mask_u_w_cutmixed2.unsqueeze(1).float(),
                ignore=ignore_unsup_2,
            )

            loss_weak_dec_w_s2 = criterion_dice(
                pred_u_s2_weak.softmax(dim=1),
                mask_u_w_weak_cutmixed2.unsqueeze(1).float(),
                ignore=ignore_weak,
            )
            loss_strong_dec_w_s2 = criterion_dice(
                pred_u_s2_strong.softmax(dim=1),
                mask_u_w_strong_cutmixed2.unsqueeze(1).float(),
                ignore=ignore_strong,
            )

            masks = mask_u_w.unsqueeze(1).float()
            loss_weak_dec_w_t = criterion_dice(
                pred_u_w_weak.softmax(dim=1),
                masks,
                ignore=ignore_teacher,
            )
            loss_strong_dec_w_t = criterion_dice(
                pred_u_w_strong.softmax(dim=1),
                masks,
                ignore=ignore_teacher,
            )
            loss_weak_dec_w_s2_t = criterion_dice(
                pred_u_s2_weak.softmax(dim=1),
                mask_u_w_cutmixed2.unsqueeze(1).float(),
                ignore=ignore_unsup_2,
            )
            loss_strong_dec_w_s2_t = criterion_dice(
                pred_u_s2_strong.softmax(dim=1),
                mask_u_w_cutmixed2.unsqueeze(1).float(),
                ignore=ignore_unsup_2,
            )

            loss_u_w_kd = (1 - args.eta) * (
                loss_weak_dec_w_t + loss_strong_dec_w_t + loss_weak_dec_w_s2_t + loss_strong_dec_w_s2_t
            ) / 4.0 + args.eta * (loss_weak_dec_w_s2 + loss_strong_dec_w_s2) / 2.0

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_kd) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_kd.update(loss_u_w_kd.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            iter_num += 1

            valid_pixels = max((ignore_mask != 255).sum().item(), 1)
            mask_ratio = ((conf_u_w >= conf_thresh) & (ignore_mask != 255)).sum().item() / valid_pixels
            total_mask_ratio.update(mask_ratio)

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
            writer.add_scalar('train/loss_kd', loss_u_w_kd.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if i % log_interval == 0:
                logger.info(
                    'Iters: {:}/{:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss KD: {:.3f}, Mask ratio: {:.3f}'.format(
                        iter_num,
                        total_iters,
                        lr,
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_loss_kd.avg,
                        total_mask_ratio.avg,
                    )
                )

        if iter_num >= total_iters * 0.3 and epoch % 2 == 0:
            model.eval()
            mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode='crossmatch')

            for cls_idx, dice in enumerate(dice_class):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info('*** Evaluation: Class [{:} {:}] Dice: {:.3f}'.format(cls_idx + 1, class_name, dice.item()))
                writer.add_scalar(f'eval/{class_name}_DICE', dice.item(), epoch)

            logger.info('*** Evaluation: MeanDice {:.3f}'.format(mDice.item()))
            writer.add_scalar('eval/mDice', mDice.item(), epoch)

            is_best = mDice.item() >= previous_best
            previous_best = max(mDice.item(), previous_best)
            if is_best:
                best_epoch = epoch

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
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
        'CrossMatch_BUSI/Ep{}bs{}_{}_seed{}_label{}/thresh{}_eta{}_{}'.format(
            cfg['epochs'],
            cfg['batch_size'],
            cfg['dataset'],
            args.seed,
            args.labelnum,
            cfg['conf_thresh'],
            args.eta,
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
