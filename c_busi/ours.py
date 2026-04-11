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
import torch.nn.functional as F
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
    'ema_decay': 0.996,
    'align_eps': 1e-8,
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
    return parser


def load_cfg(args, defaults):
    cfg = dict(DEFAULT_BUSI_CFG)
    cfg.update(defaults)
    if args.config is not None:
        with open(args.config, 'r') as f:
            cfg.update(yaml.load(f, Loader=yaml.Loader))
    cfg['dataset'] = args.dataset
    return cfg


def build_class_names(cfg):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(str(cfg.get('dataset')).upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def forward_with_bottleneck(model, x):
    feature = model.encoder(x)
    bottleneck_feature = feature[-1]
    decoder_feature = model.decoder(feature)
    preds = model.classifier(decoder_feature)
    return bottleneck_feature, preds


@torch.no_grad()
def update_ema_model(student_model, teacher_model, ema_ratio):
    for param, param_ema in zip(student_model.parameters(), teacher_model.parameters()):
        param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
    for buffer, buffer_ema in zip(student_model.buffers(), teacher_model.buffers()):
        buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    student_model = init_2d_weight(UNet(in_chns=3, class_num=cfg['nclass']).cuda()).cuda()
    teacher_model = deepcopy(student_model).cuda()
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(params=student_model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(count_params(student_model)))

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

    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d', total_iters)
    class_names = build_class_names(cfg)
    log_interval = max(len(trainloader) // 8, 1)
    pre_best_dice_student = 0.0
    pre_best_dice_teacher = 0.0
    best_epoch_student = 0
    best_epoch_teacher = 0
    epoch = -1
    iter_num = 0
    latest_path = os.path.join(cp_path, 'latest.pth')

    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        student_model.load_state_dict(checkpoint['student_model'])
        teacher_model.load_state_dict(checkpoint['teacher_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        pre_best_dice_student = checkpoint['previous_best_student']
        pre_best_dice_teacher = checkpoint['previous_best_teacher']
        best_epoch_student = checkpoint['best_epoch_student']
        best_epoch_teacher = checkpoint['best_epoch_teacher']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        start_time = time.time()

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, labelnum: {args.labelnum}, Previous best mdice '
            f'student: {pre_best_dice_student:.4f} @epoch: {best_epoch_student}, '
            f'teacher: {pre_best_dice_teacher:.4f} @epoch: {best_epoch_teacher}'
        )
        total_loss = AverageMeter()
        student_model.train()
        is_best = False

        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_x = img_x.cuda()
            mask_x = mask_x.cuda().long()

            img_w, img_s, ignore_mask, cutmix_box = unlabeled_data
            img_w = img_w.cuda()
            img_s = img_s.cuda()
            ignore_mask = ignore_mask.cuda().long()
            cutmix_box = cutmix_box.cuda().float()

            with torch.no_grad():
                teacher_logits = teacher_model(img_w).detach()
                teacher_prob = torch.softmax(teacher_logits, dim=1)
                conf_u_w, pseudo_targets_u = teacher_prob.max(dim=1)

            mask_M = 1.0 - cutmix_box.unsqueeze(1)
            images_mix = img_x * mask_M + img_s * (1.0 - mask_M)
            targets_l = mask_x

            optimizer.zero_grad(set_to_none=True)
            F_mix, preds = forward_with_bottleneck(student_model, images_mix)

            if F_mix.requires_grad and not F_mix.is_leaf:
                F_mix.retain_grad()

            # 构建监督区域与无监督区域掩码
            # mask_M: [B, 1, H, W] -> mask_M_pred: [B, 1, Hp, Wp]
            if mask_M.shape[-2:] != preds.shape[-2:]:
                mask_M_pred = F.interpolate(mask_M, size=preds.shape[-2:], mode='nearest')
            else:
                mask_M_pred = mask_M

            if ignore_mask.shape[-2:] != preds.shape[-2:]:
                ignore_mask_pred = F.interpolate(ignore_mask.unsqueeze(1).float(), size=preds.shape[-2:], mode='nearest').squeeze(1).long()
            else:
                ignore_mask_pred = ignore_mask

            if conf_u_w.shape[-2:] != preds.shape[-2:]:
                conf_u_w_pred = F.interpolate(conf_u_w.unsqueeze(1), size=preds.shape[-2:], mode='nearest').squeeze(1)
            else:
                conf_u_w_pred = conf_u_w

            if targets_l.dim() == 4 and targets_l.size(1) == 1:
                targets_l = targets_l.squeeze(1)
            if pseudo_targets_u.dim() == 4 and pseudo_targets_u.size(1) == 1:
                pseudo_targets_u = pseudo_targets_u.squeeze(1)

            if targets_l.shape[-2:] != preds.shape[-2:]:
                targets_l = F.interpolate(targets_l.unsqueeze(1).float(), size=preds.shape[-2:], mode='nearest').squeeze(1).long()
            if pseudo_targets_u.shape[-2:] != preds.shape[-2:]:
                pseudo_targets_u = F.interpolate(pseudo_targets_u.unsqueeze(1).float(), size=preds.shape[-2:], mode='nearest').squeeze(1).long()

            # 监督 CE：仅在标注区域计算
            # sup_ce_map: [B, Hp, Wp] -> loss_sup: scalar
            sup_region = mask_M_pred.squeeze(1)
            sup_ce_map = F.cross_entropy(preds, targets_l, reduction='none', ignore_index=255)
            loss_sup = (sup_ce_map * sup_region).sum() / sup_region.sum().clamp_min(1.0)

            # 无监督逐像素 CE：仅在 CutMix 的无标注区域且满足置信度/忽略掩码约束的位置计算
            # loss_unsup_pixelwise: [B, Hp, Wp], loss_unsup_scalar: scalar
            unsup_region = (1.0 - sup_region) * (ignore_mask_pred != 255).float() * (conf_u_w_pred >= cfg['conf_thresh']).float()
            loss_unsup_pixelwise = F.cross_entropy(preds, pseudo_targets_u, reduction='none') * unsup_region
            loss_unsup_scalar = loss_unsup_pixelwise.sum() / unsup_region.sum().clamp_min(1.0)

            # 特征级空间梯度对齐：对 F_mix 求监督/无监督梯度
            # G_sup/G_unsup: [B, C, Hf, Wf]
            G_sup = torch.autograd.grad(
                outputs=loss_sup,
                inputs=F_mix,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
            )[0].detach()
            G_unsup = torch.autograd.grad(
                outputs=loss_unsup_scalar,
                inputs=F_mix,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
            )[0].detach()

            # 通道维余弦相似度生成空间对齐图
            # S_align/weight_map_low_res: [B, 1, Hf, Wf]
            eps = cfg.get('align_eps', 1e-8)
            dot = (G_sup * G_unsup).sum(dim=1, keepdim=True)
            norm_sup = torch.sqrt((G_sup * G_sup).sum(dim=1, keepdim=True).clamp_min(eps))
            norm_unsup = torch.sqrt((G_unsup * G_unsup).sum(dim=1, keepdim=True).clamp_min(eps))
            S_align = dot / (norm_sup * norm_unsup + eps)
            weight_map_low_res = torch.relu(S_align)

            # 上采样到预测分辨率并重加权无监督损失
            # weight_map_high_res: [B, 1, Hp, Wp] -> weighted_loss_unsup: scalar
            weight_map_high_res = F.interpolate(
                weight_map_low_res,
                size=preds.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(1).detach()
            weighted_loss_unsup = (loss_unsup_pixelwise * weight_map_high_res).sum() / unsup_region.sum().clamp_min(1.0)

            total_loss_value = loss_sup + weighted_loss_unsup
            total_loss_value.backward()
            optimizer.step()

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            ema_ratio = min(1 - 1 / (iters + 1), cfg['ema_decay'])
            update_ema_model(student_model, teacher_model, ema_ratio)
            iter_num += 1

            total_loss.update(total_loss_value.item())
            valid_unsup_pixels = unsup_region.sum().item()
            mask_ratio = valid_unsup_pixels / max(((1.0 - sup_region) * (ignore_mask_pred != 255).float()).sum().item(), 1.0)
            align_ratio = (weight_map_high_res * unsup_region).sum().item() / max(valid_unsup_pixels, 1.0)

            writer.add_scalar('train/loss_all', total_loss_value.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_sup', loss_sup.item(), iters)
            writer.add_scalar('train/loss_unsup_raw', loss_unsup_scalar.item(), iters)
            writer.add_scalar('train/loss_unsup_weighted', weighted_loss_unsup.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            writer.add_scalar('train/align_ratio', align_ratio, iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {total_loss.avg:.3f}'
                    f', loss_sup: {loss_sup.item():.3f}, loss_u_raw: {loss_unsup_scalar.item():.3f}'
                    f', loss_u_weighted: {weighted_loss_unsup.item():.3f}, mask ratio: {mask_ratio:.4f}, align: {align_ratio:.4f}'
                )

            del G_sup, G_unsup
            if F_mix.grad is not None:
                F_mix.grad = None
            F_mix = None
            S_align = None
            weight_map_low_res = None
            weight_map_high_res = None

        if iter_num >= total_iters * 0.3 and epoch % 2 == 0:
            student_model.eval()
            mDice_student, dice_class_student = eval_2d(valloader, student_model, cfg, ifdist=False, val_mode='student')
            student_model.train()
            teacher_model.eval()
            mDice_teacher, dice_class_teacher = eval_2d(valloader, teacher_model, cfg, ifdist=False, val_mode='teacher')

            for cls_idx, dice in enumerate(dice_class_student):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info('*** Evaluation: Class [{:} {:}] Dice student: {:.3f}, teacher: {:.3f}'.format(
                    cls_idx + 1, class_name, dice, dice_class_teacher[cls_idx]
                ))
                writer.add_scalar(f'eval/{class_name}_student_DICE', dice, epoch)
                writer.add_scalar(f'eval/{class_name}_teacher_DICE', dice_class_teacher[cls_idx], epoch)

            logger.info('*** Evaluation: MeanDice student: {:.3f}, teacher: {:.3f}'.format(mDice_student, mDice_teacher))
            writer.add_scalar('eval/mDice_student', mDice_student.item(), epoch)
            writer.add_scalar('eval/mDice_teacher', mDice_teacher.item(), epoch)

            is_best = (mDice_student.item() >= pre_best_dice_student) or (mDice_teacher.item() >= pre_best_dice_teacher)
            pre_best_dice_student = max(mDice_student.item(), pre_best_dice_student)
            pre_best_dice_teacher = max(mDice_teacher.item(), pre_best_dice_teacher)
            if mDice_student.item() == pre_best_dice_student:
                best_epoch_student = epoch
            if mDice_teacher.item() == pre_best_dice_teacher:
                best_epoch_teacher = epoch

        checkpoint = {
            'student_model': student_model.state_dict(),
            'teacher_model': teacher_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_student': pre_best_dice_student,
            'previous_best_teacher': pre_best_dice_teacher,
            'best_epoch_student': best_epoch_student,
            'best_epoch_teacher': best_epoch_teacher,
            'iter_num': iter_num,
            'start_time': start_time,
        }
        torch.save(checkpoint, latest_path)
        model_ckpt = {'student_model': student_model.state_dict(), 'teacher_model': teacher_model.state_dict()}
        if is_best:
            logger.info(
                '*** best checkpoint: MeanDice student: {:.3f}, teacher: {:.3f}\n*** exp: {}'.format(
                    mDice_student, mDice_teacher, args.exp
                )
            )
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_stu_{mDice_student:.3f}_tea_{mDice_teacher:.3f}.pth'))

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
        'OURS_BUSI/Ep{}bs{}_{}_seed{}_label{}/thresh{}_ema{}_{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, cfg['conf_thresh'], cfg['ema_decay'], args.exp
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
