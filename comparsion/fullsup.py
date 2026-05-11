import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
import yaml, time, shutil, random, logging, pprint, torch, sys, argparse, gc, csv
import numpy as np
from copy import deepcopy
from torch.optim import AdamW, SGD
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.efficient import ACDCsemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet, kaiming_normal_init_weight
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import count_params, init_log, AverageMeter, DiceLoss
from utils.val import eval_2d

def get_parser(datasetname):
    cfgs = DATASET_CONFIGS[datasetname]
    parser = argparse.ArgumentParser(description=datasetname)
    parser.add_argument('--dataset', type=str, default=datasetname, choices=DATASET_CONFIGS.keys())
    # 读取数据集专属参数
    parser.add_argument('--base_dir', type=str, default=cfgs['base_dir'])
    parser.add_argument('--labelnum', type=int, default=cfgs['labelnum'], help=cfgs['label_help'])
    parser.add_argument('--num', default=cfgs['num'], type=int, help='unlabeled data number')
    parser.add_argument('--config', type=str, default=cfgs['config'], help='Path to config file. If None, auto-generated based on labelnum')
    # 通用参数
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', type=str, default="cuda:7")
    parser.add_argument('--exp', type=str, help='Experiment description')
    # parser.add_argument('--wandb_entity', type=str, default='stap1e-ucas', help='wandb entity/username (optional)')
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--normal', type=bool, help='celoss normal or something-aware')
    parser.add_argument('--checkpoint_path', type=str, default='/data/lhy_data/checkpoints_wyy')
    parser.add_argument('--deterministic', type=str, default=False)

    return parser

def main(args, cfg, save_path, cp_path):
    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    # 初始化模型
    model = UNet(in_chns=1, class_num=cfg['nclass']).cuda()
    model = kaiming_normal_init_weight(model).cuda()

    # 优化器
    optimizer = AdamW( 
        params=model.parameters(),
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)   
    
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()
    
    num_gpus = torch.cuda.device_count()
    logger.info('use {} gpus!'.format(num_gpus))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

    # === [重构 1：极简的全监督 Dataloader] ===
    trainset_l = ACDCsemiDataset('train_l', args, cfg['crop_size']) 
    valset = ACDCsemiDataset('val', args, cfg['crop_size'])
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    # 全监督只需要对有标签数据进行标准的 DataLoader 打乱采样
    trainloader = DataLoader(
        trainset_l,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        drop_last=True
    )
    # ============================================

    total_iters = len(trainloader) * cfg['epochs'] 
    print('Total iters: %d' % total_iters)
    pre_best_dice = 0.0
    best_epoch = 0
    epoch = -1
    iter_num = 0
    
    # === [重构 2：移除 EMA 的 Checkpoint 加载] ===
    if os.path.exists(os.path.join(cp_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(cp_path, 'latest.pth'), weights_only=False)
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
            
    for epoch in range(epoch + 1, cfg['epochs']):
        all_epoch, exp = cfg['epochs'], args.exp.split('_')[-1]
        logger.info(f'===> Epoch: {epoch}/{all_epoch}, {exp}, seed:{args.seed}, labelnum: {args.labelnum}, Previous best mdice '
                    f'model: {pre_best_dice:.4f} @epoch: {best_epoch}')
        total_loss  = AverageMeter()
        is_best = False
        model.train()
        
        # === [重构 3：纯净的全监督训练循环] ===
        for i, labeled_data in enumerate(trainloader):
            # 注意：标准的 DataLoader 返回的是元组，不需要解包 unlabeled_data
            img_x, mask_x = labeled_data[0].cuda(), labeled_data[1].cuda()
            
            # supervised loss
            pred_l = model(img_x)
            loss_dice = diceloss(pred_l, mask_x)
            loss_ce = criterion_l(pred_l, mask_x)
            loss_l = (loss_ce + loss_dice) / 2.0

            # 全监督中只有有监督 Loss
            loss = loss_l
            
            iter_num = iter_num + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss.update(loss.item())
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
             
            # === [TensorBoard Log] ===
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/loss_l', loss_l.item(), iters)

            # 打印日志
            if (i % max(1, len(trainloader) // 2) == 0):
                logger.info(f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {total_loss.avg:.3f}'
                            f', loss_l: {loss_l.item():.3f}, loss_dice: {loss_dice.item():.3f}, loss_ce: {loss_ce.item():.3f}')

        # === [重构 4：全监督 Validation Loop] ===
        if iter_num >= total_iters * 0.3 and epoch % 2 == 0:
            model.eval()
            mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode='model')
            model.train()        
            
            for (cls_idx, dice) in enumerate(dice_class):
                class_name = CLASSES[cfg['dataset']][cls_idx]
                logger.info('*** Evaluation: Class [{:} {:}] Dice model: {:.3f}'.format(cls_idx + 1, class_name, dice))
                writer.add_scalar('eval/%smodel_DICE' % class_name, dice, epoch)

            logger.info('*** Evaluation: MeanDice model: {:.3f}'.format(mDice))
            writer.add_scalar('eval/mDice', mDice.item(), epoch)
            
            is_best = (mDice.item() >= pre_best_dice)
            pre_best_dice = max(mDice.item(), pre_best_dice)
            if is_best:
                best_epoch = epoch

        # === [保存 Checkpoint] ===
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'epoch': epoch,
            'previous_best': pre_best_dice,
            'best_epoch': best_epoch,
            'iter_num': iter_num, 'start_time': start_time}
        torch.save(checkpoint, os.path.join(cp_path, 'latest.pth'))
        
        model_ckpt = {'model': model.state_dict()}
        if is_best:
            logger.info('*** best checkpoint: MeanDice model: {:.3f}\n*** exp: {}'.format(mDice, args.exp.split('_')[-1]))
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_m{mDice:.3f}.pth'))
            
        if epoch >= (cfg['epochs'] - 1):
            end_time = time.time()
            logger.info('Training time: {:.2f}s'.format((end_time - start_time)))
            gc.collect()
            torch.cuda.empty_cache()

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
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    cp_path = os.path.join(args.checkpoint_path, 'FullSup/Ep{}bs{}_{}_seed{}_label{}/thresh{}_{}'.format(cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, cfg['conf_thresh'], args.exp))
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
                # 如果是文件夹，使用 copytree，并可以继续使用 ignore_patterns 过滤内部垃圾文件
                shutil.copytree(s, d, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                shutil.copy2(s, d)
        else:
            print(f"Warning: {item} not found.")

    main(args, cfg, save_path, cp_path)
