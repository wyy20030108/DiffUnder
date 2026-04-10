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
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 注意：这里移除了 TwoStreamBatchSampler 和 mix_collate_fn，因为全监督不再需要双流采样
from Datasets.efficient import BUSISemiDataset 
from models.unet2d import UNet
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import count_params, init_log, AverageMeter, DiceLoss
from utils.val import eval_2d

DEFAULT_BUSI_CFG = {
    'dataset': 'BUSI',
    'base_dir': '/data/lhy_data/BUSI',
    'nclass': 2,
    'lr': 0.001,
    'epochs': 200,
    'batch_size': 8,
    'crop_size': 256,
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
    parser = argparse.ArgumentParser(description=defaults.get('dataset', 'BUSI') + ' Full Supervised')
    parser.add_argument('--dataset', type=str, default=defaults.get('dataset', 'BUSI'))
    parser.add_argument('--base_dir', type=str, default=defaults.get('base_dir', DEFAULT_BUSI_CFG['base_dir']))
    # 移除了 num, labelnum, conf_thresh 因为全监督不需要
    parser.add_argument('--config', type=str, default=defaults.get('config'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp', type=str, required=True, help='experiment name')
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


def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    # 1. 仅初始化一个模型
    model = init_2d_weight(UNet(in_chns=3, class_num=cfg['nclass']).cuda()).cuda()
    optimizer = AdamW(params=model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    
    # 2. 损失函数
    criterion_ce = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()

    logger.info('use {} gpus!'.format(torch.cuda.device_count()))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

    # 3. 数据集与 DataLoader 设置
    # 强制将 labelnum 设为 None，结合 dataset.py 逻辑，将使用 100% 的有标签训练数据
    args.labelnum = None  
    args.num = None
    
    trainset = BUSISemiDataset('train_l', args, cfg['crop_size'])
    valset = BUSISemiDataset('val', args, cfg['crop_size'])
    
    logger.info(f"Total training labeled images: {len(trainset)}")
    
    # 全监督直接使用标准 DataLoader
    trainloader = DataLoader(
        trainset,
        batch_size=cfg['batch_size'],
        shuffle=True,          # 开启打乱
        num_workers=6,
        pin_memory=True,
        drop_last=True         # 防止最后一个 batch size 为 1 导致 BN 层报错
    )
    
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    total_iters = len(trainloader) * cfg['epochs']
    logger.info('Total iters: %d', total_iters)
    class_names = build_class_names(cfg)
    log_interval = max(len(trainloader) // 8, 1)
    
    pre_best_dice = 0.0
    best_epoch = 0
    epoch = -1
    iter_num = 0
    latest_path = os.path.join(cp_path, 'latest.pth')

    # 4. 加载 Checkpoint
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        pre_best_dice = checkpoint['previous_best']
        best_epoch = checkpoint['best_epoch']
        iter_num = checkpoint['iter_num']
        start_time = checkpoint['start_time']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        start_time = time.time()

    # 5. 训练循环
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info(
            f'===> Epoch: {epoch}/{cfg["epochs"]}, seed:{args.seed}, '
            f'Previous best mdice: {pre_best_dice:.4f} @epoch: {best_epoch}'
        )
        total_loss_meter = AverageMeter()
        loss_ce_meter = AverageMeter()
        loss_dice_meter = AverageMeter()
        
        model.train()
        is_best = False

        for i, (img_x, mask_x) in enumerate(trainloader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()

            # 前向传播与损失计算
            pred = model(img_x)
            loss_ce = criterion_ce(pred, mask_x)
            loss_dice = diceloss(pred, mask_x)
            loss = (loss_ce + loss_dice) / 2.0

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_meter.update(loss.item())
            loss_ce_meter.update(loss_ce.item())
            loss_dice_meter.update(loss_dice.item())
            
            # 学习率调整 (Poly 衰减策略)
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / max(total_iters, 1)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            iter_num += 1

            # Tensorboard 记录
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/lr', lr, iters)

            if i % log_interval == 0:
                logger.info(
                    f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {total_loss_meter.avg:.3f}'
                    f', CE loss: {loss_ce_meter.avg:.3f}, Dice loss: {loss_dice_meter.avg:.3f}'
                )

        # 6. 验证循环
        # 全监督可以尽早开始验证，这里设为前 10% iters 后开始，或者每个 epoch 都验证
        if iter_num >= total_iters * 0.1 and epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode='model')
            model.train()

            for cls_idx, dice in enumerate(dice_class):
                class_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx + 1)
                logger.info('*** Evaluation: Class [{:} {:}] Dice: {:.3f}'.format(
                    cls_idx + 1, class_name, dice
                ))
                writer.add_scalar(f'eval/{class_name}_DICE', dice, epoch)

            logger.info('*** Evaluation: MeanDice: {:.3f}'.format(mDice))
            writer.add_scalar('eval/mDice', mDice.item(), epoch)

            if mDice.item() >= pre_best_dice:
                is_best = True
                pre_best_dice = mDice.item()
                best_epoch = epoch

        # 7. Checkpoint 保存
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': pre_best_dice,
            'best_epoch': best_epoch,
            'iter_num': iter_num,
            'start_time': start_time,
        }
        torch.save(checkpoint, latest_path)
        
        if is_best:
            logger.info('*** best checkpoint: MeanDice: {:.3f}\n*** exp: {}'.format(mDice, args.exp))
            model_ckpt = {'model': model.state_dict()}
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_mdice_{mDice:.3f}.pth'))

        # 清理内存
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

    # 路径修改为 FullSup_BUSI 避免与 CPS 的结果冲突
    cp_path = os.path.join(
        args.checkpoint_path,
        'FullSup_BUSI/Ep{}bs{}_{}_seed{}/{}'.format(
            cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.exp
        ),
    )
    os.makedirs(cp_path, exist_ok=True)
    save_path = os.path.join(cp_path, 'log')
    os.makedirs(save_path, exist_ok=True)

    include_list = ['Datasets', 'models', 'utils', 'configs', 'tools']
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