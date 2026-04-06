import os
import re
import sys
import h5py
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)
from Datasets.efficient import ACDCsemiDataset
from models.unet2d import UNet
from utils.classes import CLASSES


def get_parser():
    parser = argparse.ArgumentParser(description='2D test script')
    parser.add_argument('--dataset', type=str, default='ACDC')
    parser.add_argument('--base_dir', type=str, default='/data/lhy_data/ACDC')
    parser.add_argument('--labelnum', type=int, default=14)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--split', type=str, default='test', choices=['test'])
    parser.add_argument('--save_model_path', type=str, required=True)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--nclass', type=int, default=4)
    return parser


def build_fixed_cfg(args):
    return {
        'dataset': args.dataset,
        'nclass': args.nclass if args.nclass is not None else 4,
        'crop_size': [256, 256],
    }


def normalize_crop_size(crop_size):
    if isinstance(crop_size, int):
        return int(crop_size), int(crop_size)
    if isinstance(crop_size, (list, tuple)):
        if len(crop_size) == 1:
            return int(crop_size[0]), int(crop_size[0])
        if len(crop_size) >= 2:
            return int(crop_size[-2]), int(crop_size[-1])
    raise ValueError(f'Invalid crop_size: {crop_size}')


def extract_patient_id(sample_name):
    match = re.search(r'(patient\d+)', sample_name)
    if match is None:
        raise ValueError(f'Cannot parse patient id from {sample_name}')
    return match.group(1)


def build_case_groups(dataset):
    case_to_slices = {}
    for sample_name in dataset.name_list:
        patient_id = extract_patient_id(sample_name)
        case_to_slices.setdefault(patient_id, []).append(sample_name)
    return {case_id: sorted(slice_names) for case_id, slice_names in sorted(case_to_slices.items())}


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

    preferred_keys = ['model_ema', 'ema', 'teacher', 'model', 'state_dict', 'student', 'net']
    if use_ema:
        preferred_keys = ['model_ema', 'ema', 'teacher', 'model', 'state_dict', 'student', 'net']
    else:
        preferred_keys = ['model', 'state_dict', 'student', 'net', 'model_ema', 'ema', 'teacher']

    for key in preferred_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    is_raw_state_dict = all(isinstance(value, torch.Tensor) for value in checkpoint.values())
    if is_raw_state_dict:
        return checkpoint

    raise KeyError(
        'Cannot resolve state_dict from checkpoint. Available keys: {}'.format(list(checkpoint.keys()))
    )


def sanitize_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    sanitized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]
        sanitized[new_key] = value
    return sanitized


def load_model(args, cfg):
    model = UNet(in_chns=1, class_num=cfg['nclass']).cuda()
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


def infer_case(model, dataset, slice_names, crop_size):
    patch_h, patch_w = normalize_crop_size(crop_size)
    predictions = []
    labels = []
    for sample_name in slice_names:
        sample_path = dataset._get_sample_path(sample_name)
        with h5py.File(sample_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]

        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        orig_h, orig_w = image_tensor.shape[-2:]
        if (orig_h, orig_w) != (patch_h, patch_w):
            input_tensor = F.interpolate(image_tensor, size=(patch_h, patch_w), mode='bilinear', align_corners=False)
        else:
            input_tensor = image_tensor

        with torch.no_grad():
            logits = forward_logits(model, input_tensor)
            if logits.shape[-2:] != (orig_h, orig_w):
                logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        predictions.append(pred)
        labels.append(np.asarray(label, dtype=np.uint8))

    return np.stack(predictions, axis=0), np.stack(labels, axis=0)


def get_class_names(cfg):
    class_names = CLASSES.get(cfg.get('dataset')) or CLASSES.get(str(cfg.get('dataset')).upper())
    if class_names is None:
        return [str(i) for i in range(1, cfg['nclass'])]
    if len(class_names) == cfg['nclass']:
        return class_names[1:]
    return class_names[:cfg['nclass'] - 1]


def test_acdc(args, cfg):
    dataset = ACDCsemiDataset(args.split, args, cfg.get('crop_size'))
    case_groups = build_case_groups(dataset)
    model = load_model(args, cfg)
    class_names = get_class_names(cfg)

    total_metrics = np.zeros((cfg['nclass'] - 1, 4), dtype=np.float64)
    case_count = 0

    for case_id, slice_names in tqdm(case_groups.items(), ncols=100):
        prediction, label = infer_case(model, dataset, slice_names, cfg['crop_size'])
        case_metrics = []
        for class_idx in range(1, cfg['nclass']):
            case_metric = calculate_metric_percase(prediction == class_idx, label == class_idx)
            case_metrics.append(case_metric)
        case_metrics = np.asarray(case_metrics, dtype=np.float64)
        total_metrics += case_metrics
        case_count += 1

        metric_text = ', '.join([
            '{}: Dice {:.4f}, Jc {:.4f}, HD95 {:.4f}, ASD {:.4f}'.format(
                class_names[idx] if idx < len(class_names) else str(idx + 1),
                case_metrics[idx, 0],
                case_metrics[idx, 1],
                case_metrics[idx, 2],
                case_metrics[idx, 3],
            )
            for idx in range(case_metrics.shape[0])
        ])
        print(f'{case_id}: {metric_text}')

    if case_count == 0:
        raise RuntimeError(f'No cases found for split {args.split}')

    avg_metrics = total_metrics / case_count
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
            mean_metrics[0],
            mean_metrics[1],
            mean_metrics[2],
            mean_metrics[3],
        )
    )
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
    test_acdc(args, cfg)
