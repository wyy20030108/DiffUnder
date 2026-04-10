import itertools, torch, glob
import os, sys, re, random
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import h5py
from copy import deepcopy
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Datasets.transform import *

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices"""
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)

# 用于解决 ConcatDataset 中 labeled(len=2) 和 unlabeled(len=4) 返回结构不一样的问题
def mix_collate_fn(batch):
    labeled_batch = []
    unlabeled_batch = []
    
    for sample in batch:
        # 根据返回值的长度判断是 labeled 还是 unlabeled
        if len(sample) == 2:  # train_l 返回 (img, mask)
            labeled_batch.append(sample)
        elif len(sample) == 4: # train_u 返回 (img_w, img_s, ignore, box)
            unlabeled_batch.append(sample)
        else:
            raise ValueError(f"Unknown sample length: {len(sample)}")
            
    # 分别进行 collate
    return default_collate(labeled_batch), default_collate(unlabeled_batch)

class ACDCsemiDataset(Dataset):
    def __init__(self, mode, args, size=None):
        self.mode = mode
        self.args = args
        self.size = size
        self.root_path = getattr(args, 'base_dir', '/data/lhy_data/ACDC')
        self.labelnum = getattr(args, 'labelnum', None)
        self.images_h5_dir = self._resolve_images_h5_dir()
        self.all_slice_names = self._list_all_slice_names()
        self.patient_to_slices = self._build_patient_to_slices(self.all_slice_names)
        self.name_list = self._build_name_list()
        print(f"{mode} data number is: {len(self.name_list)}")

    def _resolve_images_h5_dir(self):
        for folder in ['Images_h5', 'images_h5']:
            folder_path = os.path.join(self.root_path, folder)
            if os.path.isdir(folder_path):
                return folder_path
        raise FileNotFoundError(f'Images_h5 not found under {self.root_path}')

    def _list_all_slice_names(self):
        return sorted([
            file_name[:-3]
            for file_name in os.listdir(self.images_h5_dir)
            if file_name.endswith('.h5')
        ])

    def _extract_patient_id(self, sample_name):
        match = re.search(r'(patient\d+)', sample_name)
        if match is None:
            raise ValueError(f'Cannot parse patient id from {sample_name}')
        return match.group(1)

    def _build_patient_to_slices(self, slice_names):
        patient_to_slices = {}
        for sample_name in slice_names:
            patient_id = self._extract_patient_id(sample_name)
            patient_to_slices.setdefault(patient_id, []).append(sample_name)
        for patient_id in patient_to_slices:
            patient_to_slices[patient_id] = sorted(patient_to_slices[patient_id])
        return patient_to_slices

    def _resolve_split_path(self):
        if self.mode in ['train_l', 'train_u']:
            candidates = ['train.txt', 'train_slice.txt']
        elif self.mode == 'val':
            candidates = ['val.txt', 'val_slice.txt']
        elif self.mode in ['test', 'val_test']:
            candidates = ['test.txt', 'val.txt', 'test_slice.txt', 'val_slice.txt']
        else:
            candidates = [f'{self.mode}.txt']
        for candidate in candidates:
            split_path = os.path.join(self.root_path, candidate)
            if os.path.exists(split_path):
                return split_path
        return None

    def _read_split_items(self, split_path):
        if split_path is None:
            return []
        with open(split_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _expand_items_to_slice_names(self, items):
        expanded = []
        for item in items:
            sample_name = item[:-3] if item.endswith('.h5') else item
            if sample_name in self.all_slice_names:
                expanded.append(sample_name)
                continue
            patient_id = self._extract_patient_id(sample_name)
            if patient_id in self.patient_to_slices:
                expanded.extend(self.patient_to_slices[patient_id])
        return sorted(set(expanded))

    def _build_train_slice_names(self):
        split_items = self._read_split_items(self._resolve_split_path())
        if split_items:
            expanded = self._expand_items_to_slice_names(split_items)
            if expanded:
                return expanded
        return self.all_slice_names

    def _split_train_patients(self, train_slice_names):
        patient_ids = sorted({self._extract_patient_id(name) for name in train_slice_names})
        rng = random.Random(1337)
        rng.shuffle(patient_ids)
        if self.labelnum is None:
            labelnum = len(patient_ids)
        else:
            labelnum = max(0, min(int(self.labelnum), len(patient_ids)))
        labeled_patients = set(patient_ids[:labelnum])
        unlabeled_patients = set(patient_ids[labelnum:])
        return labeled_patients, unlabeled_patients

    def _build_name_list(self):
        if self.mode in ['train_l', 'train_u']:
            train_slice_names = self._build_train_slice_names()
            labeled_patients, unlabeled_patients = self._split_train_patients(train_slice_names)
            if self.mode == 'train_l':
                return [name for name in train_slice_names if self._extract_patient_id(name) in labeled_patients]
            return [name for name in train_slice_names if self._extract_patient_id(name) in unlabeled_patients]
        split_items = self._read_split_items(self._resolve_split_path())
        if split_items:
            expanded = self._expand_items_to_slice_names(split_items)
            if expanded:
                return expanded
        return self.all_slice_names

    def _get_sample_path(self, sample_name):
        return os.path.join(self.images_h5_dir, f'{sample_name}.h5')

    def _to_uint8_image(self, image):
        image = np.asarray(image, dtype=np.float32)
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image, dtype=np.float32)
        return np.uint8(np.clip(image * 255.0, 0, 255))

    def _to_tensor_image(self, image):
        image = np.asarray(image, dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return torch.from_numpy(image).float().unsqueeze(0)

    def _to_tensor_mask(self, mask):
        return torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()).long()

    def _normalize_size(self):
        size_cfg = getattr(self.args, 'crop_size', None)
        size_value = self.size if size_cfg is None else size_cfg
        if size_value is None:
            return None
        if isinstance(size_value, int):
            return size_value
        if isinstance(size_value, (list, tuple)):
            if len(size_value) == 0:
                return None
            return int(size_value[0])
        return int(size_value)

    def _apply_train_aug(self, image, mask, ignore_value):
        crop_size = self._normalize_size()
        if crop_size is not None:
            image, mask = crop(image, mask, crop_size, ignore_value=ignore_value)
        image, mask = hflip(image, mask)
        return image, mask

    def __getitem__(self, idx):
        sample_name = self.name_list[idx]
        sample_path = self._get_sample_path(sample_name)
        with h5py.File(sample_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:] if ('label' in h5f and self.mode != 'train_u') else np.zeros_like(image, dtype=np.uint8)

        image_pil = Image.fromarray(self._to_uint8_image(image))
        mask_pil = Image.fromarray(np.asarray(label, dtype=np.uint8))

        if self.mode == 'train_l':
            image_pil, mask_pil = self._apply_train_aug(image_pil, mask_pil, ignore_value=255)
            return self._to_tensor_image(image_pil), self._to_tensor_mask(mask_pil)

        if self.mode == 'train_u':
            image_w, mask_w = self._apply_train_aug(image_pil, mask_pil, ignore_value=254)
            image_s = blur(deepcopy(image_w), p=0.5)
            ignore_mask = np.zeros(np.asarray(mask_w).shape, dtype=np.uint8)
            ignore_mask[np.asarray(mask_w) == 254] = 255
            cutmix_box = obtain_cutmix_box(min(image_s.size))
            return (
                self._to_tensor_image(image_w),
                self._to_tensor_image(image_s),
                self._to_tensor_mask(ignore_mask),
                cutmix_box
            )

        if self.mode in ['val', 'test', 'val_test']:
            return self._to_tensor_image(image), self._to_tensor_mask(label)

        raise ValueError(f'Unsupported mode: {self.mode}')

    def __len__(self):
        return len(self.name_list)


class BUSISemiDataset(Dataset):
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    MASK_DIR_NAMES = {'mask', 'masks', 'label', 'labels', 'gt', 'gts', 'annotation', 'annotations'}
    MASK_SUFFIXES = ('_mask', '-mask', '_seg', '-seg', '_label', '-label', '_gt', '-gt')

    def __init__(self, mode, args, size=None):
        self.mode = mode
        self.args = args
        self.size = size
        self.root_path = getattr(args, 'base_dir', '/data/lhy_data/BUSI')
        self.labelnum = getattr(args, 'labelnum', None)
        self.num = getattr(args, 'num', None)
        self.ratio_range = self._normalize_ratio_range(
            getattr(args, 'ratio_range', getattr(args, 'scale_range', (0.8, 1.2)))
        )
        self.all_records = self._scan_records()
        self.record_by_relpath = {record['rel_path']: record for record in self.all_records}
        self.record_by_stem = {record['stem']: record for record in self.all_records}
        self.name_list = self._build_name_list()
        print(f"{mode} data number is: {len(self.name_list)}")

    def _normalize_ratio_range(self, ratio_range):
        if ratio_range is None:
            return None
        if isinstance(ratio_range, (int, float)):
            ratio = float(ratio_range)
            return (ratio, ratio)
        if isinstance(ratio_range, (list, tuple)) and len(ratio_range) == 2:
            return (float(ratio_range[0]), float(ratio_range[1]))
        raise ValueError(f'Invalid ratio_range: {ratio_range}')

    def _normalize_size(self):
        size_cfg = getattr(self.args, 'crop_size', None)
        size_value = self.size if size_cfg is None else size_cfg
        if size_value is None:
            return None
        if isinstance(size_value, int):
            return size_value
        if isinstance(size_value, (list, tuple)):
            if len(size_value) == 0:
                return None
            if len(size_value) == 1:
                return int(size_value[0])
            if int(size_value[0]) != int(size_value[1]):
                raise ValueError(f'BUSISemiDataset expects square crop size, but got {size_value}')
            return int(size_value[0])
        return int(size_value)

    def _canonical_relpath(self, path):
        return os.path.normpath(path).replace('\\', '/')

    def _is_image_file(self, file_name):
        return os.path.splitext(file_name)[1].lower() in self.IMAGE_EXTENSIONS

    def _is_mask_file(self, abs_path):
        parent_name = os.path.basename(os.path.dirname(abs_path)).lower()
        if parent_name in self.MASK_DIR_NAMES:
            return True
        stem = os.path.splitext(os.path.basename(abs_path))[0].lower()
        return any(suffix in stem for suffix in self.MASK_SUFFIXES)
    
    def _resolve_mask_path(self, image_path):
        """返回一个包含所有匹配 mask 路径的列表"""
        image_dir = os.path.dirname(image_path)
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        
        # 匹配该原图对应的所有 mask，例如 benign (4)_mask.png, benign (4)_mask_1.png
        # 这种写法不仅兼容 BUSI 原版命名，也兼容很多其他变体
        search_pattern = os.path.join(image_dir, f"{image_stem}*mask*.*")
        mask_candidates = glob.glob(search_pattern)
        
        # 过滤掉非图像文件，并确保不把原图自己加进去
        valid_mask_paths = [p for p in mask_candidates if self._is_image_file(p) and p != image_path]
        
        return valid_mask_paths if valid_mask_paths else None

    def _make_record(self, image_path, mask_paths=None):
        image_path = os.path.abspath(image_path)
        return {
            'image_path': image_path,
            'mask_path': [os.path.abspath(p) for p in mask_paths] if mask_paths else None,
            'rel_path': self._canonical_relpath(os.path.relpath(image_path, self.root_path)),
            'stem': os.path.splitext(os.path.basename(image_path))[0],
        }

    def _scan_records(self):
        records = []
        for current_root, _, file_names in os.walk(self.root_path):
            for file_name in sorted(file_names):
                if not self._is_image_file(file_name):
                    continue
                image_path = os.path.join(current_root, file_name)
                if self._is_mask_file(image_path):
                    continue
                records.append(self._make_record(image_path, self._resolve_mask_path(image_path)))
        if not records:
            raise FileNotFoundError(f'No BUSI images found under {self.root_path}')
        records.sort(key=lambda item: item['rel_path'])
        return records

    def _resolve_split_path(self):
        if self.mode == 'train_l':
            candidates = ['train_l.txt', 'labeled.txt', 'train.txt']
        elif self.mode == 'train_u':
            candidates = ['train_u.txt', 'unlabeled.txt', 'train.txt']
        elif self.mode == 'val':
            candidates = ['val.txt']
        elif self.mode in ['test', 'val_test']:
            candidates = ['test.txt', 'val.txt']
        else:
            candidates = [f'{self.mode}.txt']
        for candidate in candidates:
            split_path = os.path.join(self.root_path, candidate)
            if os.path.isfile(split_path):
                return split_path
        return None

    def _parse_split_line(self, line):
        line = line.strip()
        if not line:
            return None, None
        if '\t' in line:
            parts = [part.strip() for part in line.split('\t') if part.strip()]
        elif ',' in line:
            parts = [part.strip() for part in line.split(',', 1) if part.strip()]
        else:
            parts = [line]
        if len(parts) == 1:
            return parts[0], None
        return parts[0], parts[1]

    def _resolve_to_abs_path(self, path):
        if path is None:
            return None
        path = path.strip().strip('"').strip("'")
        if not path:
            return None
        if os.path.isabs(path):
            candidate = path
        else:
            candidate = os.path.join(self.root_path, path)
        candidate = os.path.abspath(os.path.normpath(candidate))
        return candidate if os.path.exists(candidate) else None

    def _lookup_record(self, image_key):
        image_key = image_key.strip().strip('"').strip("'")
        rel_key = self._canonical_relpath(image_key)
        if rel_key in self.record_by_relpath:
            return dict(self.record_by_relpath[rel_key])

        abs_path = self._resolve_to_abs_path(image_key)
        if abs_path is not None:
            abs_rel_key = self._canonical_relpath(os.path.relpath(abs_path, self.root_path))
            if abs_rel_key in self.record_by_relpath:
                return dict(self.record_by_relpath[abs_rel_key])

        stem = os.path.splitext(os.path.basename(image_key))[0]
        if stem in self.record_by_stem:
            return dict(self.record_by_stem[stem])
        return None

    def _load_records_from_split(self, split_path):
        if split_path is None:
            return []
        records = []
        with open(split_path, 'r') as f:
            for raw_line in f:
                image_key, mask_key = self._parse_split_line(raw_line)
                if image_key is None:
                    continue
                record = self._lookup_record(image_key)
                if record is None:
                    image_path = self._resolve_to_abs_path(image_key)
                    if image_path is None:
                        raise FileNotFoundError(f'Cannot resolve BUSI image path from split item: {image_key}')
                    record = self._make_record(image_path, self._resolve_mask_path(image_path))
                if mask_key is not None:
                    mask_path = self._resolve_to_abs_path(mask_key)
                    if mask_path is None:
                        raise FileNotFoundError(f'Cannot resolve BUSI mask path from split item: {mask_key}')
                    record['mask_path'] = [mask_path]
                records.append(record)
        return records

    def _split_train_records(self, train_records):
        shuffled = list(train_records)
        rng = random.Random(1337)
        rng.shuffle(shuffled)
        if self.labelnum is None:
            labelnum = len(shuffled)
        else:
            labelnum = max(0, min(int(self.labelnum), len(shuffled)))
        labeled_records = shuffled[:labelnum]
        unlabeled_records = shuffled[labelnum:]
        if self.num is not None:
            unlabeled_records = unlabeled_records[:max(0, int(self.num))]
        return labeled_records, unlabeled_records

    def _build_name_list(self):
        split_path = self._resolve_split_path()
        split_records = self._load_records_from_split(split_path)

        if self.mode == 'train_l':
            explicit_split = split_path is not None and os.path.basename(split_path) in {'train_l.txt', 'labeled.txt'}
            if explicit_split:
                return split_records
            labeled_records, _ = self._split_train_records(split_records if split_records else self.all_records)
            return labeled_records

        if self.mode == 'train_u':
            explicit_split = split_path is not None and os.path.basename(split_path) in {'train_u.txt', 'unlabeled.txt'}
            if explicit_split:
                records = split_records
                if self.num is not None:
                    records = records[:max(0, int(self.num))]
                return records
            _, unlabeled_records = self._split_train_records(split_records if split_records else self.all_records)
            return unlabeled_records

        if split_records:
            return split_records
        return self.all_records

    def _load_image(self, image_path):
        with Image.open(image_path) as img:
            return img.convert('RGB')

    def _normalize_real_mask(self, mask):
        mask = np.asarray(mask, dtype=np.uint8).copy()
        mask[mask == 255] = 1
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        return mask

    def _load_mask(self, mask_paths):
        if not mask_paths:
            raise FileNotFoundError('BUSI labeled sample requires at least one valid mask path')
        
        combined_mask = None
        
        # 遍历读取该原图对应的所有 mask
        for path in mask_paths:
            with Image.open(path) as mask_img:
                mask = self._normalize_real_mask(mask_img.convert('L'))
                
            # 将多张 mask 按位或 (逻辑融合)
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = np.maximum(combined_mask, mask)
                
        return Image.fromarray(combined_mask)

    def _apply_train_aug(self, image, mask):
        crop_size = self._normalize_size()
        if crop_size is not None:
            image = image.resize((crop_size, crop_size), Image.BILINEAR)
            mask = mask.resize((crop_size, crop_size), Image.NEAREST)
        image, mask = hflip(image, mask)
        return image, mask

    def _build_empty_mask(self, image):
        width, height = image.size
        return Image.fromarray(np.zeros((height, width), dtype=np.uint8))

    def __getitem__(self, idx):
        sample = self.name_list[idx]
        image = self._load_image(sample['image_path'])
        
        # ======== 最关键的拦截：在试图加载 Mask 之前进行安全检查 ========
        if self.mode != 'train_u' and not sample['mask_path']:
            raise FileNotFoundError(
                f"🚨 致命错误：模型在尝试读取有标签数据时，找不到对应的 Mask！\n"
                f"原图路径: {sample['image_path']}\n"
                f"请检查该目录下是否存在以 '{sample['stem']}_' 开头的掩码文件。"
            )
        # ================================================================
        
        mask = self._build_empty_mask(image) if self.mode == 'train_u' else self._load_mask(sample['mask_path'])

        if self.mode == 'train_l':
            image, mask = self._apply_train_aug(image, mask)
            return normalize(image, mask)

        if self.mode == 'train_u':
            image_w, mask_w = self._apply_train_aug(image, mask)
            image_s = blur(deepcopy(image_w), p=0.5)
            ignore_mask = np.zeros(np.asarray(mask_w).shape, dtype=np.uint8)
            ignore_mask[np.asarray(mask_w) == 254] = 255
            cutmix_box = obtain_cutmix_box(min(image_s.size))
            return (
                normalize(image_w),
                normalize(image_s),
                torch.from_numpy(ignore_mask.copy()).long(),
                cutmix_box
            )

        # 验证集/测试集的返回
        return normalize(image, mask)

    def __len__(self):
        return len(self.name_list)
    
