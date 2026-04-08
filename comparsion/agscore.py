import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128')
import yaml, time, shutil, random, logging, pprint, torch, sys, argparse, gc, csv
import numpy as np
import warnings
from copy import deepcopy
from torch.optim import AdamW, SGD
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.efficient import ACDCsemiDataset, TwoStreamBatchSampler, mix_collate_fn
from models.unet2d import UNet, kaiming_normal_init_weight
from utils.classes import CLASSES
from utils.datasets import DATASET_CONFIGS
from utils.util import count_params, init_log, AverageMeter, DiceLoss
from utils.val import eval_2d

class AgScoreLabelFilter:
    def __init__(self, num_pos_agents=64, num_neg_agents=256, reduced_size=64, subsample_size=8192, gmm_grid_size=512):
        """
        初始化 AgScore 伪标签过滤器 (ICML 2025)
        参数:
            num_pos_agents (N): 正代理数量，论文默认 64
            num_neg_agents (M): 负代理数量，论文默认 256
        """
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
            mode='nearest'
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

    def _orthogonal_selection(self, candidates, M):
        """
        Algorithm 2: 负代理的正交选择策略 (Orthogonal Selection)
        选择与已选代理集具有最小最大余弦相似度的代表性负特征。
        """
        num_cands = candidates.size(0)
        if num_cands == 0:
            return None

        cands_norm = F.normalize(candidates, p=2, dim=1)
        
        # 1. 随机初始化第一个代理
        start_idx = torch.randint(0, num_cands, (1,)).item()
        selected_feats = [candidates[start_idx]]
        selected_norm = [cands_norm[start_idx]]
        
        avail_mask = torch.ones(num_cands, dtype=torch.bool, device=candidates.device)
        avail_mask[start_idx] = False
        
        # 2. 迭代选择剩余的 M-1 个代理 (Min-Max 策略)
        for _ in range(M - 1):
            if not avail_mask.any():
                # 候选耗尽时复制最后的特征
                selected_feats.append(selected_feats[-1])
                continue
                
            curr_selected_norm = torch.stack(selected_norm) # (i, C)
            avail_indices = torch.nonzero(avail_mask).squeeze(1)
            avail_cands_norm = cands_norm[avail_indices]    # (num_avail, C)
            
            # 计算相似度矩阵 -> (num_avail, i)
            sim_matrix = torch.mm(avail_cands_norm, curr_selected_norm.t())
            max_sim, _ = torch.max(sim_matrix, dim=1)
            
            # 选出最大相似度中最小的候选
            best_idx_relative = torch.argmin(max_sim)
            best_idx_absolute = avail_indices[best_idx_relative]
            
            selected_feats.append(candidates[best_idx_absolute])
            selected_norm.append(cands_norm[best_idx_absolute])
            avail_mask[best_idx_absolute] = False
            
        return torch.stack(selected_feats)

    def _build_agents(self, features, confidences, pseudo_labels):
        """
        Algorithm 1: Agent Construction (构建正代理和负代理)
        """
        B, C, H, W = features.shape
        feats_flat = features.permute(0, 2, 3, 1).reshape(-1, C) # (B*H*W, C)
        conf_flat = confidences.reshape(-1)
        pl_flat = pseudo_labels.reshape(-1)
        
        unique_classes = torch.unique(pl_flat)
        pos_candidates_dict = {}
        neg_candidates_list = []
        
        # 遍历所有类别收集 Top-1% (Positive) 和 Bottom-1% (Negative)
        for k in unique_classes:
            k_mask = (pl_flat == k)
            k_confs = conf_flat[k_mask]
            k_feats = feats_flat[k_mask]
            
            num_k_pixels = k_confs.size(0)
            k_count = max(1, int(num_k_pixels * 0.01)) # 1% 的阈值数量
            
            sorted_indices = torch.argsort(k_confs)
            # 收集候选
            neg_candidates_list.append(k_feats[sorted_indices[:k_count]])
            pos_candidates_dict[k.item()] = k_feats[sorted_indices[-k_count:]]
            
        neg_candidates = torch.cat(neg_candidates_list, dim=0)
        
        # 1. 类别平衡地构建正代理 A_p (N)
        A_p = []
        num_classes = len(pos_candidates_dict)
        if num_classes > 0:
            samples_per_class = self.N // num_classes
            remainder = self.N % num_classes
            for i, (k, cands) in enumerate(pos_candidates_dict.items()):
                num_to_sample = samples_per_class + (1 if i < remainder else 0)
                if cands.size(0) >= num_to_sample:
                    idx = torch.randperm(cands.size(0), device=cands.device)[:num_to_sample]
                else: # 有放回重采样
                    idx = torch.randint(0, cands.size(0), (num_to_sample,), device=cands.device)
                A_p.append(cands[idx])
            A_p = torch.cat(A_p, dim=0)
            
        # 2. 通过正交策略构建负代理 A_n (M)
        A_n = self._orthogonal_selection(neg_candidates, self.M)
        
        # 处理异常边界情况（如果本 Batch 置信度过低导致挑选失败）
        if A_p is None or A_n is None or len(A_p) == 0:
            return None, None
            
        return A_p, A_n

    def _calculate_agscore(self, features, A_p, A_n):
        """ Equation 4: 计算 AgScore """
        B, C, H, W = features.shape
        feats_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        feats_norm = F.normalize(feats_flat, p=2, dim=1)
        Ap_norm = F.normalize(A_p, p=2, dim=1)
        An_norm = F.normalize(A_n, p=2, dim=1)
        
        # 计算与正代理/负代理的相似度的指数和
        sum_exp_pos = torch.sum(torch.exp(torch.mm(feats_norm, Ap_norm.t())), dim=1)
        sum_exp_neg = torch.sum(torch.exp(torch.mm(feats_norm, An_norm.t())), dim=1)
        
        ag_score = sum_exp_pos / (sum_exp_pos + sum_exp_neg)
        return ag_score.reshape(B, H, W)

    def get_pseudo_label_mask(self, features, confidences, pseudo_labels, fallback_threshold=0.95):
        """ 主调用接口 """
        B, H, W = pseudo_labels.shape
        reduced_features, reduced_confidences, reduced_pseudo_labels, original_size = self._reduce_inputs(
            features, confidences, pseudo_labels
        )
        
        # 1. 构建代理
        A_p, A_n = self._build_agents(reduced_features, reduced_confidences, reduced_pseudo_labels)
        if A_p is None or A_n is None:
            # 安全回退机制：退化为固定的 confidence 阈值
            return confidences >= fallback_threshold
            
        # 2. 计算 AgScore
        ag_scores = self._calculate_agscore(reduced_features, A_p, A_n)
        
        # 3. GMM 模型拟合过滤 (DivideMix 逻辑)
        scores_np = ag_scores.detach().cpu().numpy().reshape(-1, 1)
        if scores_np.shape[0] == 0:
            return confidences >= fallback_threshold
        if not np.isfinite(scores_np).all():
            return confidences >= fallback_threshold
        
        # 【重要提速优化】: 随机采样 5 万个像素来 fit GMM，避免 100 万级像素 fit 卡死训练
        subsample_size = self.subsample_size
        if scores_np.shape[0] > subsample_size:
            sampled_indices = np.random.choice(scores_np.shape[0], subsample_size, replace=False)
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
            if any(issubclass(w.category, ConvergenceWarning) for w in caught_warnings) or (hasattr(gmm, 'converged_') and not gmm.converged_):
                return confidences >= fallback_threshold
            correct_idx = np.argmax(gmm.means_)
            threshold = self._solve_threshold(gmm, fit_scores, correct_idx)
            if threshold is None:
                return confidences >= fallback_threshold
            valid_mask = ag_scores >= threshold
            if original_size is not None:
                valid_mask = F.interpolate(
                    valid_mask.unsqueeze(1).float(),
                    size=original_size,
                    mode='nearest'
                ).squeeze(1) > 0.5
            return valid_mask.reshape(B, H, W)
        except Exception:
            # 防止 GMM 极端情况下崩溃
            return confidences >= fallback_threshold
        
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
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--val_num_workers', type=int, default=1)
    parser.add_argument('--agscore_reduced_size', type=int, default=64)
    parser.add_argument('--agscore_subsample_size', type=int, default=8192)
    parser.add_argument('--agscore_gmm_grid_size', type=int, default=512)

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

    model = UNet(in_chns=1, class_num=cfg['nclass']).cuda()
    model = kaiming_normal_init_weight(model).cuda()
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
        
    optimizer = AdamW( 
        params=model.parameters(),
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)   
    # optimizer = SGD(model.parameters(), lr=cfg['lr'],
    #                       momentum=0.9, weight_decay=0.0001)  # 0.0001
    
    criterion_l = nn.CrossEntropyLoss(ignore_index=255).cuda()
    diceloss = DiceLoss(cfg['nclass']).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()
    
    num_gpus = torch.cuda.device_count()
    logger.info('use {} gpus!'.format(num_gpus))
    logger.info('Total params: {:.3f}M'.format(count_params(model)))

    # 1. 创建 Dataset 实例
    trainset_u = ACDCsemiDataset('train_u', args, cfg['crop_size'])
    trainset_l = ACDCsemiDataset('train_l', args, cfg['crop_size']) 

    valset = ACDCsemiDataset('val', args, cfg['crop_size'])
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=args.val_num_workers, drop_last=False)

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
        num_workers=args.num_workers,
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
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
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
        all_epoch, exp = cfg['epochs'], args.exp.split('_')[-1]
        logger.info(f'===> Epoch: {epoch}/{all_epoch}, {exp}, seed:{args.seed}, labelnum: {args.labelnum}, Previous best mdice '
                    f'model: {pre_best_dice:.4f} @epoch: {best_epoch}, ema: {pre_best_dice_ema:.4f} @epoch_ema: {best_epoch_ema}')
        total_loss  = AverageMeter()
        is_best = False
        model.train()
        
        for i, (labeled_data, unlabeled_data) in enumerate(trainloader):
            img_x, mask_x = labeled_data
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_w, img_s, ignore_mask, _ = unlabeled_data
            img_w, img_s = img_w.cuda(), img_s.cuda()
            ignore_mask = ignore_mask.cuda()
            
            # supervised loss
            pred_l = model(img_x)
            loss_dice = diceloss(pred_l, mask_x)
            loss_ce = criterion_l(pred_l, mask_x)
            loss_l = (loss_ce + loss_dice) / 2.0

            # 1. Teacher 模型获取预测和【像素级特征】
            pred_u_w, feature_u_w = model_ema(img_w, use_feature=True)
            pred_u_w = pred_u_w.detach()
            feature_u_w = feature_u_w.detach()

            pred_u_w_soft = torch.softmax(pred_u_w, dim=1)
            conf_u_w, mask_u_w = pred_u_w_soft.max(dim=1)
            
            # 2. 替代原来的 pred_w_high_confident_mask = conf_u_w >= cfg['conf_thresh']
            # 使用 AgScore 产出高质量有效 Mask
            ag_valid_mask = ag_filter.get_pseudo_label_mask(
                feature_u_w, conf_u_w, mask_u_w, fallback_threshold=cfg['conf_thresh']
            )
            
            # 3. 结合 Ignore Mask (剔除边界/无效标注区)
            selected_mask_u = ag_valid_mask & (ignore_mask != 255)
                
            # 4. Student 模型前向与计算损失
            pred_u_student_s = model(img_s)
            loss_u = criterion_u(pred_u_student_s, mask_u_w)
            loss_u = loss_u * selected_mask_u
            
            # 避免除以 0 的异常保护
            valid_pixel_count = selected_mask_u.sum().item()
            if valid_pixel_count > 0:
                loss_u = loss_u.sum() / valid_pixel_count
            else:
                loss_u = loss_u.sum() * 0.0

            # 统计通过 AgScore 验证的像素比例 (用于记录 Tensorboard)
            mask_ratio = valid_pixel_count / (ignore_mask != 255).sum().item()
            loss = loss_l + loss_u * 0.1
            
            iter_num = iter_num + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss.update(loss.item())
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)

            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
             
            # === [TensorBoard Log] ===
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/lr', lr, iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
            writer.add_scalar('train/loss_l', loss_l.item(), iters)
            writer.add_scalar('train/loss_u', loss_u.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
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
                            f', loss_l: {loss_l.item():.3f}, loss_dice: {loss_dice.item():.3f}, loss_ce: {loss_ce.item():.3f}'
                            f', loss_u: {loss_u.item():.3f}, mask ratio: {mask_ratio:.4f}')

        # Validation Loop
        if iter_num >= total_iters * 0.3 and epoch % 2 == 0:
        # if iter_num >= total_iters * 0:
            model.eval()
            mDice, dice_class = eval_2d(valloader, model, cfg, ifdist=False, val_mode='model')
            model.train()        
            mDice_ema, dice_class_ema = eval_2d(valloader, model_ema, cfg, ifdist=False, val_mode='ema')           
            
            # # === [WandB & TensorBoard Log: Validation] ===
            # wandb_val_log = {}
            
            for (cls_idx, dice) in enumerate(dice_class):
                class_name = CLASSES[cfg['dataset']][cls_idx]
                logger.info('*** Evaluation: Class [{:} {:}] Dice model: {:.3f}, ema: {:.3f}'.format(cls_idx + 1, class_name, dice, dice_class_ema[cls_idx]))
                
                # TensorBoard, WandB
                writer.add_scalar('eval/%smodel_DICE' % class_name, dice, epoch)
                writer.add_scalar('eval/%smodel_ema_DICE' % class_name, dice_class_ema[cls_idx], epoch)
                # wandb_val_log[f'eval/{class_name}_model_DICE'] = dice
                # wandb_val_log[f'eval/{class_name}_model_ema_DICE'] = dice_class_ema[cls_idx]

            logger.info('*** Evaluation:  MeanDice model: {:.3f}, ema: {:.3f}'.format(mDice, mDice_ema))
            
            # TensorBoard, WandB
            writer.add_scalar('eval/mDice', mDice.item(), epoch)
            writer.add_scalar('eval/mDICE_ema', mDice_ema.item(), epoch)
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
            'model': model.state_dict(),  'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(), 'epoch': epoch,
            'previous_best': pre_best_dice,'previous_best_ema': pre_best_dice_ema,
            'best_epoch': best_epoch, 'best_epoch_ema': best_epoch_ema,
            'iter_num': iter_num, 'start_time': start_time}
        torch.save(checkpoint, os.path.join(cp_path, 'latest.pth'))
        model_ckpt = {'model': model.state_dict(), 'model_ema': model_ema.state_dict()}
        if is_best:
            logger.info('*** best checkpoint:  MeanDice model: {:.3f}, ema: {:.3f}\n*** exp: {}'.format(mDice, mDice_ema, args.exp.split('_')[-1]))
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
    
    cp_path = os.path.join(args.checkpoint_path, 'AgScore/Ep{}bs{}_{}_seed{}_label{}/thresh{}_{}'.format(cfg['epochs'], cfg['batch_size'], cfg['dataset'], args.seed, args.labelnum, cfg['conf_thresh'], args.exp))
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
