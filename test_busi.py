import os
import sys
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from medpy import metric
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
from Datasets.efficient import BUSISemiDataset
from models.unet2d import UNet
from utils.classes import CLASSES

def get_parser():
    parser = argparse.ArgumentParser(description='BUSI test script')
    parser.add_argument('--dataset', type=str, default='BUSI')
    parser.add_argument('--base_dir', type=str, default='/data/lhy_data/BUSI')
    parser.add_argument('--labelnum', type=int, default=100)
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'val_test'])
    parser.add_argument('--save_model_path', type=str, required=True)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--strict_eval_masks', action='store_true', default=False)
    parser.add_argument('--save_pred_root', type=str, default='/data/lhy_data/checkpoints_wyy/test')
    return parser


def build_fixed_cfg(args):
    return {
        'dataset': args.dataset,
        'nclass': args.nclass if args.nclass is not None else 2,
        'crop_size': args.crop_size,
    }


def normalize_eval_size(crop_size):
    if crop_size is None:
        return None
    if isinstance(crop_size, int):
        return int(crop_size), int(crop_size)
    if isinstance(crop_size, (list, tuple)):
        if len(crop_size) == 0:
            return None
        if len(crop_size) == 1:
            return int(crop_size[0]), int(crop_size[0])
        return int(crop_size[0]), int(crop_size[1])
    return int(crop_size), int(crop_size)


def calculate_metric_percase(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0, 0.0, 0.0
    return 0.0, 0.0, 50.0, 10.0


def resolve_state_dict(checkpoint, use_ema=False):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if use_ema:
        preferred_keys = ['model_ema', 'ema', 'teacher', 'model2', 'model', 'state_dict', 'student', 'net', 'teacher_model']
    else:
        preferred_keys = ['model', 'model1', 'state_dict', 'student', 'net', 'model_ema', 'ema', 'teacher', 'model2', 'student_model']
    for key in preferred_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    is_raw_state_dict = all(isinstance(value, torch.Tensor) for value in checkpoint.values())
    if is_raw_state_dict:
        return checkpoint
    raise KeyError('Cannot resolve state_dict from checkpoint. Available keys: {}'.format(list(checkpoint.keys())))


def sanitize_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    sanitized = {}
    for key, value in state_dict.items():
        new_key = key[len('module.'):] if key.startswith('module.') else key
        sanitized[new_key] = value
    return sanitized


def load_model(args, cfg):
    model = UNet(in_chns=3, class_num=cfg['nclass']).cuda()
    checkpoint = torch.load(args.save_model_path, map_location='cpu', weights_only=False)
    state_dict = sanitize_state_dict(resolve_state_dict(checkpoint, use_ema=args.use_ema))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def forward_logits(model, input_tensor):
    output = model(input_tensor)
    if isinstance(output, dict):
        for key in ['out', 'pred', 'logits']:
            if key in output:
                return output[key]
        raise KeyError(f'Unknown output keys: {list(output.keys())}')
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def get_class_names(cfg):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(str(cfg.get('dataset')).upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def build_overlay_palette():
    return np.asarray(
        [
            [255, 64, 64],
            [64, 200, 255],
            [255, 196, 64],
            [128, 255, 128],
            [255, 128, 255],
            [128, 128, 255],
        ],
        dtype=np.uint8,
    )


def build_output_paths(sample_record, save_pred_root):
    rel_path = sample_record.get('rel_path') or os.path.basename(sample_record['image_path'])
    rel_root, _ = os.path.splitext(rel_path)
    visual_path = os.path.join(save_pred_root, 'visuals', rel_root + '.png')
    return visual_path


def colorize_mask(mask, cfg):
    palette = build_overlay_palette()
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(1, cfg['nclass']):
        class_mask = mask == class_idx
        if not np.any(class_mask):
            continue
        color_mask[class_mask] = palette[(class_idx - 1) % len(palette)]
    return color_mask


def add_title_bar(panels, titles, spacer_width=8, title_height=28):
    panel_width = panels[0].shape[1]
    canvas_width = panel_width * len(panels) + spacer_width * (len(panels) - 1)
    title_bar = Image.new('RGB', (canvas_width, title_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(title_bar)
    font = ImageFont.load_default()
    for idx, title in enumerate(titles):
        x0 = idx * (panel_width + spacer_width)
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = x0 + max(0, (panel_width - text_w) // 2)
        text_y = max(0, (title_height - text_h) // 2 - text_bbox[1])
        draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)
    return np.asarray(title_bar, dtype=np.uint8)


def save_prediction_visuals(sample_record, pred, gt, args, cfg):
    if not args.save_pred_root:
        return
    visual_path = build_output_paths(sample_record, args.save_pred_root)
    os.makedirs(os.path.dirname(visual_path), exist_ok=True)

    with Image.open(sample_record['image_path']) as image:
        base_image = image.convert('RGB')
    target_size = (pred.shape[1], pred.shape[0])
    if base_image.size != target_size:
        base_image = base_image.resize(target_size, Image.BILINEAR)

    image_array = np.asarray(base_image, dtype=np.uint8)
    pred_overlay = image_array.astype(np.float32)
    pred_color = colorize_mask(pred.astype(np.uint8), cfg).astype(np.float32)
    pred_foreground = pred > 0
    pred_overlay[pred_foreground] = 0.5 * pred_overlay[pred_foreground] + 0.5 * pred_color[pred_foreground]
    pred_overlay = np.clip(pred_overlay, 0, 255).astype(np.uint8)

    gt_overlay = image_array.astype(np.float32)
    gt_color = colorize_mask(gt.astype(np.uint8), cfg).astype(np.float32)
    gt_foreground = gt > 0
    gt_overlay[gt_foreground] = 0.5 * gt_overlay[gt_foreground] + 0.5 * gt_color[gt_foreground]
    gt_overlay = np.clip(gt_overlay, 0, 255).astype(np.uint8)

    spacer = np.full((pred.shape[0], 8, 3), 255, dtype=np.uint8)
    merged_body = np.concatenate([image_array, spacer, pred_overlay, spacer, gt_overlay], axis=1)
    title_bar = add_title_bar([image_array, pred_overlay, gt_overlay], ['Img', 'Img+Pred', 'Img+GT'])
    merged_image = np.concatenate([title_bar, merged_body], axis=0)
    Image.fromarray(merged_image).save(visual_path)


def test_busi(args, cfg):
    dataset = BUSISemiDataset(args.split, args, cfg.get('crop_size'))
    model = load_model(args, cfg)
    class_names = get_class_names(cfg)
    total_metrics = np.zeros((cfg['nclass'] - 1, 4), dtype=np.float64)
    sample_count = 0
    saved_visual_count = 0
    valid_indices = []
    missing_indices = []
    for idx, sample in enumerate(dataset.name_list):
        if isinstance(sample, dict):
            mask_paths = sample.get('mask_path')
            if mask_paths:
                valid_indices.append(idx)
            else:
                missing_indices.append(idx)
        else:
            valid_indices.append(idx)
    if missing_indices:
        print(f'Skip {len(missing_indices)} samples without mask in split "{args.split}".')
        if args.strict_eval_masks:
            raise FileNotFoundError('Found samples without mask while strict_eval_masks is enabled.')
    if not valid_indices:
        raise RuntimeError(f'No valid labeled samples found for split {args.split}.')
    if args.save_pred_root:
        os.makedirs(args.save_pred_root, exist_ok=True)

    eval_size = normalize_eval_size(cfg.get('crop_size'))
    for idx in tqdm(valid_indices, ncols=100):
        sample_record = dataset.name_list[idx]
        image, label = dataset[idx]
        image = image.unsqueeze(0).cuda()
        label = label.numpy().astype(np.uint8)
        orig_h, orig_w = image.shape[-2:]
        if eval_size is not None and (orig_h != eval_size[0] or orig_w != eval_size[1]):
            input_tensor = F.interpolate(image, size=eval_size, mode='bilinear', align_corners=False)
        else:
            input_tensor = image
        with torch.no_grad():
            logits = forward_logits(model, input_tensor)
            if logits.shape[-2:] != (orig_h, orig_w):
                logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        sample_metrics = []
        for class_idx in range(1, cfg['nclass']):
            sample_metrics.append(calculate_metric_percase(pred == class_idx, label == class_idx))
        sample_metrics = np.asarray(sample_metrics, dtype=np.float64)
        total_metrics += sample_metrics
        sample_count += 1
        if args.save_pred_root:
            save_prediction_visuals(sample_record, pred, label, args, cfg)
            saved_visual_count += 1

    if sample_count == 0:
        raise RuntimeError(f'No samples found for split {args.split}')

    avg_metrics = total_metrics / sample_count
    print('\nAverage metrics per class:')
    for idx, metrics_per_class in enumerate(avg_metrics):
        class_name = class_names[idx] if idx < len(class_names) else str(idx + 1)
        print(
            '{} -> Dice: {:.4f}, Jaccard: {:.4f}, HD95: {:.4f}, ASD: {:.4f}'.format(
                class_name,
                metrics_per_class[0],
                metrics_per_class[1],
                metrics_per_class[2],
                metrics_per_class[3],
            )
        )

    mean_metrics = avg_metrics.mean(axis=0)
    print(
        '\nMean metrics -> Dice: {:.4f}, Jaccard: {:.4f}, HD95: {:.4f}, ASD: {:.4f}'.format(
            mean_metrics[0], mean_metrics[1], mean_metrics[2], mean_metrics[3]
        )
    )
    if args.save_pred_root:
        print(f'Saved {saved_visual_count} merged visualization images to {args.save_pred_root}')
    return avg_metrics, mean_metrics


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    cfg = build_fixed_cfg(args)
    test_busi(args, cfg)
