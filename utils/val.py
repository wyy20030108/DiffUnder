import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from copy import deepcopy

def evaluate_3d(loader, model, cfg, ifdist=None, val_mode=None, ifdycon=False, gnet=False, mgnet=False, decouple_classifier=False, majority_map=None, minority_map=None, lab7=False):
    print(f"{val_mode} Validation begin")
    model.eval()
    total_samples = 0
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.no_grad(): 
            all_mDice_organ = 0
            for img, mask in tqdm(loader, ncols=100):
                img, mask, add_pad = img.cuda(), mask.squeeze(0).cuda(), False
                if total_samples==0:
                    dice_class_all = torch.zeros((cfg['nclass']-1,), device=img.device)
                total_samples += 1
                dice_class = torch.zeros((cfg['nclass']-1,), device=img.device) # dismiss background
                b, c, d, h, w = img.shape
                patch_d, patch_h, patch_w = cfg['val_patch_size']
                d_pad = max(0, patch_d - d)
                h_pad = max(0, patch_h - h)
                w_pad = max(0, patch_w - w)
                if d_pad > 0 or h_pad > 0 or w_pad > 0:
                    add_pad = True
                wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
                hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
                dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
                padding_tuple = (wl_pad, wr_pad, hl_pad, hr_pad, dl_pad, dr_pad)
                if any(padding_tuple):
                    img = F.pad(img, padding_tuple, mode='constant', value=0)
                    mask = F.pad(mask, padding_tuple, mode='constant', value=0)
                d, h, w = img.shape[2:]
                score_map = torch.zeros((cfg['nclass'], ) + torch.Size([d, h, w]), device=img.device)
                count_map = torch.zeros(img.shape[2:], device=img.device)
    
                num_h = math.ceil(h / patch_h)
                overlap_h = (patch_h * num_h - h) // (num_h - 1) if num_h > 1 else 0
                stride_h = patch_h - overlap_h
                num_w = math.ceil(w / patch_w)
                overlap_w = (patch_w * num_w - w) // (num_w - 1) if num_w > 1 else 0
                stride_w = patch_w - overlap_w
                num_d = math.ceil(d / patch_d)
                overlap_d = (patch_d * num_d - d) // (num_d - 1) if num_d > 1 else 0
                stride_d = patch_d - overlap_d
                d_starts = torch.arange(0, num_d, device=img.device) * stride_d
                d_starts = torch.clamp(d_starts, max=d-patch_d)
                h_starts = torch.arange(0, num_h, device=img.device) * stride_h
                h_starts = torch.clamp(h_starts, max=h-patch_h)
                w_starts = torch.arange(0, num_w, device=img.device) * stride_w
                w_starts = torch.clamp(w_starts, max=w-patch_w)
                d_idx = d_starts[:, None] + torch.arange(patch_d, device=img.device)
                h_idx = h_starts[:, None] + torch.arange(patch_h, device=img.device)   
                w_idx = w_starts[:, None] + torch.arange(patch_w, device=img.device)  

                for dd in d_idx:
                    for hh in h_idx:
                        for ww in w_idx:
                            d_id = dd.unsqueeze(1).unsqueeze(2).expand(-1, patch_h, patch_w)
                            h_id = hh.unsqueeze(0).unsqueeze(2).expand(patch_d, -1, patch_w)  
                            w_id = ww.unsqueeze(0).unsqueeze(1).expand(patch_d, patch_h, -1)    
                            input = img[:, :, d_id, h_id, w_id]
                            if ifdycon:
                                _, pred_patch_logits, _ = model(input)
                            elif gnet:
                                pred_patch_logits = model(input)["pred"]
                            elif mgnet:
                                pred_patch_logits = model(input)[0]
                            elif decouple_classifier:
                                pred_majority, pred_minority = model(input)
                                B, _, H, W, D = pred_majority.shape
                                pred_patch_logits = torch.full((B, 14, H, W, D), -1e9, device=pred_majority.device)
                                for orig_id, new_id in majority_map.items():
                                    pred_patch_logits[:, orig_id, :, :] = pred_majority[:, new_id, :, :]
                                for orig_id, new_id in minority_map.items():
                                    pred_patch_logits[:, orig_id, :, :] = pred_minority[:, new_id, :, :]
                            elif lab7: 
                                pred_origin = model(input)
                                debiased_input = deepcopy(input)
                                debiased_input[:, :, :, :, :] = input.max()
                                pred_most = model(debiased_input)
                                pred_patch_logits = pred_origin - pred_most
                                pred_patch_logits_background = pred_origin.argmax(dim=1)==0
                                pred_patch_logits_unbiased_foreground = pred_patch_logits.argmax(dim=1)!=0
                                pred_patch_logits_background[pred_patch_logits_unbiased_foreground==True] = False 
                                pred_patch_logits = pred_patch_logits.permute(0, 2, 3, 4, 1)
                                pred_patch_logits[pred_patch_logits_background, :] = pred_origin.permute(0, 2, 3, 4, 1)[pred_patch_logits_background, :]
                                pred_patch_logits = pred_patch_logits.permute(0, 4, 1, 2, 3)
                            else:
                                pred_patch_logits = model(input)
                            # pred_patch_logits = model(input)
                            pred_patch = torch.softmax(pred_patch_logits, dim=1)
                            score_map[:, d_id, h_id, w_id] += pred_patch.squeeze(0)
                            count_map[d_id, h_id, w_id] += 1

                if add_pad:
                    score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
                    count_map = count_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
                score_map /= count_map
                pred_mask = score_map.argmax(dim=0)
                del score_map, count_map, input, pred_patch_logits, pred_patch
                torch.cuda.empty_cache()

                classes = torch.arange(1, cfg['nclass'], device=mask.device) # (nclass-1, 1, 1, 1)
                mask_exp = mask.unsqueeze(0) == classes.unsqueeze(1).unsqueeze(2).unsqueeze(3) # (nclass-1, D, H, W)
                pred_exp = pred_mask.unsqueeze(0) == classes.unsqueeze(1).unsqueeze(2).unsqueeze(3) # (nclass-1, D, H, W)
                mask_exp, pred_exp = mask_exp.view(cfg['nclass']-1, -1), pred_exp.view(cfg['nclass']-1, -1)
                intersection = (mask_exp * pred_exp).sum(dim=1)
                union = mask_exp.sum(dim=1) + pred_exp.sum(dim=1)
                del mask_exp, pred_exp
                torch.cuda.empty_cache()
                
                dice_class += (2. * intersection) / (union + 1e-7)
                mDice_organ = dice_class.mean()
                all_mDice_organ += mDice_organ
                dice_class_all += dice_class
                del intersection, union, pred_mask
                torch.cuda.empty_cache()
            total_samples_tensor = torch.tensor(total_samples).cuda()
    if ifdist:
        dist.all_reduce(all_mDice_organ)
        dist.all_reduce(dice_class_all)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        all_mDice_organ /= total_samples_tensor
        dice_class_all /= total_samples_tensor
    else:
        all_mDice_organ /= total_samples_tensor
        dice_class_all /= total_samples_tensor

    print(f"{val_mode} Validation end")
    return all_mDice_organ, dice_class_all


def eval_2d(loader, model, cfg, ifdist=None, val_mode=None, ifdycon=False, gnet=False, mgnet=False, decouple_classifier=False, majority_map=None, minority_map=None, lab7=False):
    print(f"{val_mode} Validation begin")
    model.eval()
    total_samples = 0
    stream = torch.cuda.Stream()

    def _forward_model(input_patch):
        if ifdycon:
            _, pred_patch_logits, _ = model(input_patch)
        elif gnet:
            pred_patch_logits = model(input_patch)["pred"]
        elif mgnet:
            pred_patch_logits = model(input_patch)[0]
        elif decouple_classifier:
            pred_majority, pred_minority = model(input_patch)
            b, _, h, w = pred_majority.shape
            pred_patch_logits = torch.full((b, cfg['nclass'], h, w), -1e9, device=pred_majority.device)
            for orig_id, new_id in majority_map.items():
                pred_patch_logits[:, orig_id, :, :] = pred_majority[:, new_id, :, :]
            for orig_id, new_id in minority_map.items():
                pred_patch_logits[:, orig_id, :, :] = pred_minority[:, new_id, :, :]
        elif lab7:
            pred_origin = model(input_patch)
            debiased_input = torch.full_like(input_patch, input_patch.max())
            pred_most = model(debiased_input)
            pred_patch_logits = pred_origin - pred_most
            pred_bg = pred_origin.argmax(dim=1) == 0
            pred_fg_unbiased = pred_patch_logits.argmax(dim=1) != 0
            pred_bg[pred_fg_unbiased] = False
            pred_patch_logits = pred_patch_logits.permute(0, 2, 3, 1)
            pred_patch_logits[pred_bg, :] = pred_origin.permute(0, 2, 3, 1)[pred_bg, :]
            pred_patch_logits = pred_patch_logits.permute(0, 3, 1, 2)
        else:
            pred_patch_logits = model(input_patch)
        if isinstance(pred_patch_logits, (tuple, list)):
            pred_patch_logits = pred_patch_logits[0]
        if isinstance(pred_patch_logits, dict):
            pred_patch_logits = pred_patch_logits["pred"]
        return pred_patch_logits

    with torch.cuda.stream(stream):
        with torch.no_grad():
            all_mDice_organ = 0
            for img, mask in tqdm(loader, ncols=100):
                img = img.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                if mask.ndim == 4 and mask.shape[1] == 1:
                    mask = mask[:, 0]
                mask = mask.long()
                if total_samples == 0:
                    dice_class_all = torch.zeros((cfg['nclass'] - 1,), device=img.device)
                bs, _, h, w = img.shape
                crop_size = cfg.get('crop_size', cfg.get('val_patch_size'))
                if isinstance(crop_size, int):
                    patch_h = crop_size
                    patch_w = crop_size
                elif isinstance(crop_size, (list, tuple)):
                    if len(crop_size) == 1:
                        patch_h = int(crop_size[0])
                        patch_w = int(crop_size[0])
                    elif len(crop_size) >= 2:
                        patch_h = int(crop_size[-2])
                        patch_w = int(crop_size[-1])
                    else:
                        raise ValueError('crop_size is empty')
                else:
                    raise ValueError('crop_size or val_patch_size must be provided for eval_2d')

                h_pad = max(0, patch_h - h)
                w_pad = max(0, patch_w - w)
                add_pad = h_pad > 0 or w_pad > 0
                hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
                wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
                padding_tuple = (wl_pad, wr_pad, hl_pad, hr_pad)
                if add_pad:
                    img = F.pad(img, padding_tuple, mode='constant', value=0)
                    mask = F.pad(mask, padding_tuple, mode='constant', value=0)

                padded_h, padded_w = img.shape[2:]
                score_map = torch.zeros((bs, cfg['nclass'], padded_h, padded_w), device=img.device)
                count_map = torch.zeros((bs, 1, padded_h, padded_w), device=img.device)

                num_h = math.ceil(padded_h / patch_h)
                overlap_h = (patch_h * num_h - padded_h) // (num_h - 1) if num_h > 1 else 0
                stride_h = patch_h - overlap_h
                num_w = math.ceil(padded_w / patch_w)
                overlap_w = (patch_w * num_w - padded_w) // (num_w - 1) if num_w > 1 else 0
                stride_w = patch_w - overlap_w

                h_starts = torch.arange(0, num_h, device=img.device) * stride_h
                h_starts = torch.clamp(h_starts, max=padded_h - patch_h)
                w_starts = torch.arange(0, num_w, device=img.device) * stride_w
                w_starts = torch.clamp(w_starts, max=padded_w - patch_w)

                for h_start in h_starts.tolist():
                    for w_start in w_starts.tolist():
                        input_patch = img[:, :, h_start:h_start + patch_h, w_start:w_start + patch_w]
                        pred_patch = torch.softmax(_forward_model(input_patch), dim=1)
                        score_map[:, :, h_start:h_start + patch_h, w_start:w_start + patch_w] += pred_patch
                        count_map[:, :, h_start:h_start + patch_h, w_start:w_start + patch_w] += 1

                if add_pad:
                    score_map = score_map[:, :, hl_pad:hl_pad + h, wl_pad:wl_pad + w]
                    count_map = count_map[:, :, hl_pad:hl_pad + h, wl_pad:wl_pad + w]
                    mask = mask[:, hl_pad:hl_pad + h, wl_pad:wl_pad + w]

                score_map = score_map / count_map.clamp_min(1.0)
                pred_mask = score_map.argmax(dim=1)
                classes = torch.arange(1, cfg['nclass'], device=img.device).view(1, -1, 1, 1)
                mask_exp = (mask.unsqueeze(1) == classes).view(mask.shape[0], cfg['nclass'] - 1, -1)
                pred_exp = (pred_mask.unsqueeze(1) == classes).view(mask.shape[0], cfg['nclass'] - 1, -1)
                intersection = (mask_exp & pred_exp).sum(dim=2).float()
                union = mask_exp.sum(dim=2).float() + pred_exp.sum(dim=2).float()
                dice_class = (2.0 * intersection) / (union + 1e-7)
                dice_class_all += dice_class.sum(dim=0)
                all_mDice_organ += dice_class.mean(dim=1).sum()
                total_samples += bs

                del score_map, count_map, pred_mask, mask_exp, pred_exp, intersection, union
                torch.cuda.empty_cache()

            total_samples_tensor = torch.tensor(total_samples, device=torch.device('cuda'), dtype=torch.float32)
    if ifdist:
        dist.all_reduce(all_mDice_organ)
        dist.all_reduce(dice_class_all)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    total_samples_tensor = torch.clamp(total_samples_tensor, min=1.0)
    all_mDice_organ = all_mDice_organ / total_samples_tensor
    dice_class_all = dice_class_all / total_samples_tensor
    print(f"{val_mode} Validation end")
    return all_mDice_organ, dice_class_all


def evaluate_3d_denoising(loader, model, cfg, ifdist=None, val_mode=None):
    """
    用于 3D 图像去噪/修复的验证函数
    """
    print(f"{val_mode} Denoising Validation begin")
    model.eval()
    total_samples = 0
    
    # 初始化指标累加器
    total_mse_bg = 0.0  # 背景区域的均方误差
    total_mae_bg = 0.0  # 背景区域的平均绝对误差
    total_voxels_bg = 0 # 背景区域的总像素数

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.no_grad():
            for img, mask in tqdm(loader):
                # img: [1, C, D, H, W], mask: [1, 1, D, H, W]
                input_img, mask = img.cuda(), mask.cuda()
                img_mask = deepcopy(img)
                img_mask = img_mask.squeeze(0).cuda()
                img_mask[mask.expand_as(img_mask)>0] = img.min() # 前景作为噪声, 应该去除
                
                b, c, d, h, w = img.shape
                patch_d, patch_h, patch_w = cfg['val_patch_size']
                
                # --- Padding 逻辑 (保持不变) ---
                d_pad = max(0, patch_d - d)
                h_pad = max(0, patch_h - h)
                w_pad = max(0, patch_w - w)
                add_pad = False
                if d_pad > 0 or h_pad > 0 or w_pad > 0:
                    add_pad = True
                wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
                hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
                dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
                padding_tuple = (wl_pad, wr_pad, hl_pad, hr_pad, dl_pad, dr_pad)
                
                if any(padding_tuple):
                    input_img = F.pad(input_img, padding_tuple, mode='constant', value=0)
                    # 原始 img 和 mask 不需要 pad 进网络，但为了后续取出对应的 crop，这里也 pad 一下方便处理
                    # 或者我们只 pad input，输出后再 crop 回来。为了逻辑简单，这里同步 pad。
                    img_padded = F.pad(img, padding_tuple, mode='constant', value=0)
                    mask_padded = F.pad(mask, padding_tuple, mode='constant', value=0)
                else:
                    img_padded = img
                    mask_padded = mask

                # --- 准备滑动窗口容器 ---
                # score_map 存储的是重建的像素值，维度是 [in_chns, D, H, W]
                in_chns = img.shape[1] 
                pad_d, pad_h, pad_w = img_padded.shape[2:]
                recon_map = torch.zeros((in_chns, pad_d, pad_h, pad_w), device=input_img.device)
                count_map = torch.zeros((1, pad_d, pad_h, pad_w), device=input_img.device)

                # --- 滑动窗口索引生成 ---
                num_h = math.ceil(pad_h / patch_h)
                overlap_h = (patch_h * num_h - pad_h) // (num_h - 1) if num_h > 1 else 0
                stride_h = patch_h - overlap_h
                
                num_w = math.ceil(pad_w / patch_w)
                overlap_w = (patch_w * num_w - pad_w) // (num_w - 1) if num_w > 1 else 0
                stride_w = patch_w - overlap_w
                
                num_d = math.ceil(pad_d / patch_d)
                overlap_d = (patch_d * num_d - pad_d) // (num_d - 1) if num_d > 1 else 0
                stride_d = patch_d - overlap_d

                d_starts = torch.clamp(torch.arange(0, num_d, device=input_img.device) * stride_d, max=pad_d-patch_d)
                h_starts = torch.clamp(torch.arange(0, num_h, device=input_img.device) * stride_h, max=pad_h-patch_h)
                w_starts = torch.clamp(torch.arange(0, num_w, device=input_img.device) * stride_w, max=pad_w-patch_w)

                d_idx = d_starts[:, None] + torch.arange(patch_d, device=input_img.device)
                h_idx = h_starts[:, None] + torch.arange(patch_h, device=input_img.device)
                w_idx = w_starts[:, None] + torch.arange(patch_w, device=input_img.device)

                # --- 滑动推理 ---
                for dd in d_idx:
                    for hh in h_idx:
                        for ww in w_idx:
                            # 提取 Patch 维度扩展用于切片
                            d_id = dd.unsqueeze(1).unsqueeze(2).expand(-1, patch_h, patch_w)
                            h_id = hh.unsqueeze(0).unsqueeze(2).expand(patch_d, -1, patch_w)
                            w_id = ww.unsqueeze(0).unsqueeze(1).expand(patch_d, patch_h, -1)

                            input_patch = input_img[:, :, d_id, h_id, w_id] # [B, C, D, H, W]

                            # 模型推理 这里的 model 返回 (seg, recon)，我们需要 recon
                            out = model(input_patch)
                            if isinstance(out, tuple) or isinstance(out, list):
                                # 假设 recon_out 是第二个返回值
                                recon_patch = out[1] 
                            else:
                                recon_patch = out

                            # 累加结果
                            recon_map[:, d_id, h_id, w_id] += recon_patch.squeeze(0)
                            count_map[:, d_id, h_id, w_id] += 1

                # --- 结果聚合 ---
                recon_map /= count_map

                # 去除 Padding，还原回原始尺寸
                if add_pad:
                    recon_map = recon_map[:, dl_pad:dl_pad + d, hl_pad:hl_pad + h, wl_pad:wl_pad + w]
               
                # 计算 MSE 和 MAE
                mse = F.mse_loss(recon_map, img_mask, reduction='sum')
                mae = F.l1_loss(recon_map, img_mask, reduction='sum')
                
                total_mse_bg += mse.item()
                total_mae_bg += mae.item()
                total_voxels_bg += recon_map.numel()
                total_samples += 1

    # --- 分布式聚合 (Distributed Reduce) ---
    total_metrics = torch.tensor([total_mse_bg, total_mae_bg, total_voxels_bg], device=input_img.device)
    
    if ifdist:
        dist.all_reduce(total_metrics, op=dist.ReduceOp.SUM)
    
    final_mse = total_metrics[0] / total_metrics[2]
    final_mae = total_metrics[1] / total_metrics[2]

    # PSNR 计算 (基于 MSE，假设数据大致在 0-1 之间或已归一化，仅供参考)
    # 如果数据是 Z-score 归一化的，PSNR 的物理意义会减弱，但 MSE 越小越好是肯定的。
    final_psnr = 10 * torch.log10(1.0 / final_mse) if final_mse > 0 else 100
    # print(f"{val_mode} Results - MSE(BG): {final_mse:.4f}, MAE(BG): {final_mae:.4f}, PSNR(BG): {final_psnr:.2f}")
    print(f"{val_mode} Validation end")
    return final_mse, final_psnr
