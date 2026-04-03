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

    # # === [WandB: Initialization] ===
    # # 初始化 wandb
    # wandb.init(
    #     project=f"lab9-seed{args.seed}-num{args.labelnum}-debackground",
    #     entity=args.wandb_entity,
    #     name=args.exp,
    #     config=all_args,
    #     dir=save_path  # 缓存目录
    # )
    # # ========================================

    model1 = UNet(in_chns=1, class_num=cfg['nclass']).cuda()
    model1 = kaiming_normal_init_weight(model1).cuda()
    
    model2 = deepcopy(model1)
    model2 = kaiming_normal_init_weight(model2).cuda()
    
    optimizer1 = AdamW( 
        params=model1.parameters(),
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)   
    optimizer2 = AdamW( 
        params=model2.parameters(),
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)   
    # optimizer = SGD(model.parameters(), lr=cfg['lr'],
    #                       momentum=0.9, weight_decay=0.0001)  # 0.0001
    
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    
    num_gpus = torch.cuda.device_count()
    logger.info('use {} gpus!'.format(num_gpus))
    logger.info('Total params: {:.3f}M'.format(count_params(model1) + count_params(model2)))

    # 1. 创建 Dataset 实例
    trainset_u = ACDCsemiDataset('train_u', args, cfg['crop_size'])
    trainset_l = ACDCsemiDataset('train_l', args, cfg['crop_size']) 

    valset = ACDCsemiDataset('val', args, cfg['crop_size'])
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    # 2. 定义 Batch Size 分配
    labeled_bs = cfg['batch_size']
    unlabeled_bs = cfg['batch_size'] 
    total_batch_size = labeled_bs + unlabeled_bs

    # 3. 创建 ConcatDataset 和 索引列表
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
    # ... (Dataset Setup 结束) ...

    total_iters = len(trainloader) * cfg['epochs'] 
    print('Total iters: %d' % total_iters)
    pre_best_dice = 0.0
    pre_best_dice_ema = 0.0
    best_epoch = 0
    best_epoch_ema = 0
    epoch = -1
    iter_num = 0
    
    if os.path.exists(os.path.join(cp_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(cp_path, 'latest.pth'), weights_only=False)
        model1.load_state_dict(checkpoint['model1'])
        model2.load_state_dict(checkpoint['model2'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
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
        all_epoch, exp = cfg['epochs'], args.exp.split('_')[-1]
        logger.info(f'===> Epoch: {epoch}/{all_epoch}, {exp}, seed:{args.seed}, labelnum: {args.labelnum}, Previous best mdice '
                    f'model: {pre_best_dice:.4f} @epoch: {best_epoch}, ema: {pre_best_dice_ema:.4f} @epoch_ema: {best_epoch_ema}')
        total_loss  = AverageMeter()
        is_best = False
        model1.train()
        model2.train()
        
        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_w, img_s, ignore_mask, _ = unlabeled_data
            img_w, img_s = img_w.cuda(), img_s.cuda()
            ignore_mask = ignore_mask.cuda()
            
            # supervised loss for both models
            pred_l1 = model1(img_x)
            loss_dice1 = diceloss(pred_l1, mask_x)
            loss_ce1 = criterion_l(pred_l1, mask_x)
            loss_l1 = (loss_ce1 + loss_dice1) / 2.0

            pred_l2 = model2(img_x)
            loss_dice2 = diceloss(pred_l2, mask_x)
            loss_ce2 = criterion_l(pred_l2, mask_x)
            loss_l2 = (loss_ce2 + loss_dice2) / 2.0

            # cross pseudo supervision
            pred_u_w1 = model1(img_w).detach()
            pred_u_w1_soft = torch.softmax(pred_u_w1, dim=1).detach()
            conf_u_w1, mask_u_w1 = pred_u_w1_soft.max(dim=1)
            pred_w_high_confident_mask1 = conf_u_w1 >= cfg['conf_thresh']
            selected_mask_u1 = pred_w_high_confident_mask1 & (ignore_mask != 255)
                
            pred_u_w2 = model2(img_w).detach()
            pred_u_w2_soft = torch.softmax(pred_u_w2, dim=1).detach()
            conf_u_w2, mask_u_w2 = pred_u_w2_soft.max(dim=1)
            pred_w_high_confident_mask2 = conf_u_w2 >= cfg['conf_thresh']
            selected_mask_u2 = pred_w_high_confident_mask2 & (ignore_mask != 255)

            # model1 learns from model2's pseudo labels
            pred_u_student_s1 = model1(img_s)
            loss_u1 = criterion_u(pred_u_student_s1, mask_u_w2)
            loss_u1 = loss_u1 * (selected_mask_u2 & (ignore_mask != 255))
            loss_u1 = loss_u1.sum() / (ignore_mask != 255).sum().item()

            # model2 learns from model1's pseudo labels
            pred_u_student_s2 = model2(img_s)
            loss_u2 = criterion_u(pred_u_student_s2, mask_u_w1)
            loss_u2 = loss_u2 * (selected_mask_u1 & (ignore_mask != 255))
            loss_u2 = loss_u2.sum() / (ignore_mask != 255).sum().item()

            # average losses
            loss_l = (loss_l1 + loss_l2) / 2.0
            loss_u = (loss_u1 + loss_u2) / 2.0
            loss = loss_l + loss_u * 0.1
            
            iter_num = iter_num + 1
            
            # update both models
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
    
            total_loss.update(loss.item())
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer1.param_groups[0]["lr"] = lr
            optimizer2.param_groups[0]["lr"] = lr
             
            # === [TensorBoard Log] ===
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_ce1', loss_ce1.item(), iters)
            writer.add_scalar('train/loss_dice1', loss_dice1.item(), iters)
            writer.add_scalar('train/loss_ce2', loss_ce2.item(), iters)
            writer.add_scalar('train/loss_dice2', loss_dice2.item(), iters)
            writer.add_scalar('train/loss_l', loss_l.item(), iters)
            writer.add_scalar('train/loss_u', loss_u.item(), iters)
            writer.add_scalar('train/loss_u1', loss_u1.item(), iters)
            writer.add_scalar('train/loss_u2', loss_u2.item(), iters)
            
            # # === [WandB Log: Training] ===
            # wandb.log({
            #     'train/loss_all': loss.item(),
            #     'train/lr': lr,
            #     'train/loss_ce': loss_ce.item(),
            #     'train/loss_dice': loss_dice.item(),
            #     'train/loss_l': loss_l.item(),
            #     'train/loss_u': loss_u.item(),
            #     'train/mask_ratio': mask_ratio.item(),
            # }, step=iters)
            # # ==============================

            if (i % (args.num // 2) == 0):
                logger.info(f'Iters: {iter_num}/{total_iters}, LR: {lr:.7f}, Total loss: {total_loss.avg:.3f}'
                            f', loss_l: {loss_l.item():.3f}, loss_u: {loss_u.item():.3f}')

        # Validation Loop
        if iter_num >= total_iters * 0.3 and epoch % 2 == 0:
        # if iter_num >= total_iters * 0:
            model1.eval()
            mDice, dice_class = eval_2d(valloader, model1, cfg, ifdist=False, val_mode='model1')
            model1.train()        
            model2.eval()
            mDice_ema, dice_class_ema = eval_2d(valloader, model2, cfg, ifdist=False, val_mode='model2')  
            model2.train()         
            
            # # === [WandB & TensorBoard Log: Validation] ===
            # wandb_val_log = {}
            
            for (cls_idx, dice) in enumerate(dice_class):
                class_name = CLASSES[cfg['dataset']][cls_idx]
                logger.info('*** Evaluation: Class [{:} {:}] Dice model1: {:.3f}, model2: {:.3f}'.format(cls_idx + 1, class_name, dice, dice_class_ema[cls_idx]))
                
                # TensorBoard, WandB
                writer.add_scalar('eval/%smodel1_DICE' % class_name, dice, epoch)
                writer.add_scalar('eval/%smodel2_DICE' % class_name, dice_class_ema[cls_idx], epoch)
                # wandb_val_log[f'eval/{class_name}_model_DICE'] = dice
                # wandb_val_log[f'eval/{class_name}_model_ema_DICE'] = dice_class_ema[cls_idx]

            logger.info('*** Evaluation:  MeanDice model1: {:.3f}, model2: {:.3f}'.format(mDice, mDice_ema))
            
            # TensorBoard, WandB
            writer.add_scalar('eval/mDice_model1', mDice.item(), epoch)
            writer.add_scalar('eval/mDice_model2', mDice_ema.item(), epoch)
            # wandb_val_log['eval/mDice'] = mDice.item()
            # wandb_val_log['eval/mDICE_ema'] = mDice_ema.item()
            # 
            # wandb.log(wandb_val_log, step=iters)  # 使用 iters 作为 step，以便与 Training Loss 对齐时间轴
            # # ===============================================
            
            is_best = (mDice.item() >= pre_best_dice) or (mDice_ema.item() >= pre_best_dice_ema)
            
            pre_best_dice = max(mDice.item(), pre_best_dice)
            pre_best_dice_ema = max(mDice_ema.item(), pre_best_dice_ema)
            if mDice.item() == pre_best_dice:
                best_epoch = epoch
            if mDice_ema.item() == pre_best_dice_ema:
                best_epoch_ema = epoch

        checkpoint = {
            'model1': model1.state_dict(),  'model2': model2.state_dict(),
            'optimizer1': optimizer1.state_dict(), 'optimizer2': optimizer2.state_dict(), 'epoch': epoch,
            'previous_best': pre_best_dice,'previous_best_ema': pre_best_dice_ema,
            'best_epoch': best_epoch, 'best_epoch_ema': best_epoch_ema,
            'iter_num': iter_num, 'start_time': start_time}
        torch.save(checkpoint, os.path.join(cp_path, 'latest.pth'))
        model_ckpt = {'model1': model1.state_dict(), 'model2': model2.state_dict()}
        if is_best:
            logger.info('*** best checkpoint:  MeanDice model1: {:.3f}, model2: {:.3f}\n*** exp: {}'.format(mDice, mDice_ema, args.exp.split('_')[-1]))
            torch.save(model_ckpt, os.path.join(cp_path, f'ep{epoch}_m{mDice:.3f}_ema{mDice_ema:.3f}.pth'))
            
        if epoch >= (cfg['epochs'] - 1):
            end_time = time.time()
            logger.info('Training time: {:.2f}s'.format((end_time - start_time)))
            gc.collect()
            torch.cuda.empty_cache()
            # wandb.finish() # 结束 wandb run

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
    
    cp_path = os.path.join(args.checkpoint_path, 'CPS/Ep{}bs{}_{}_seed{}_label{}/thresh{}_{}'.format(cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, cfg['conf_thresh'], args.exp))
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
