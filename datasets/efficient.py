import itertools, torch
import os
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import h5py
from copy import deepcopy
from PIL import Image

from datasets.transform import *

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



class Flare_fixmatch_Dataset_effi(Dataset):
    """ Flare2022 Dataset with cutmix """
    def __init__(self, mode, args, size):
        self.dir = args.base_dir
        self.size = size
        self.mode = mode
        if mode == 'val_test':
            self.path = self.dir + f'/val.txt'
        else:
            self.path = self.dir + f'/{mode}.txt' 
        with open(self.path, 'r') as f:
            self.name_list = f.readlines()
        self.name_list = [item.replace('\n', '') for item in self.name_list]
        
        if mode == 'train_u' and args.num is not None:
            self.name_list = self.name_list[:args.num]
        
        if mode == 'train_l' and args.labelnum is not None:
            self.name_list = self.name_list[:args.labelnum]
        
        print(f"{mode} data number is: {len(self.name_list)}")

    def __getitem__(self, idx):
        id = self.name_list[idx]
        if self.mode.split('_')[0] == 'train':
            if self.mode == 'train_u':
                flag = 'unlabeled'
            elif self.mode == 'train_l':
                flag = 'labeled'
            else:
                raise ValueError(f"self.mode: {self.mode} is error, must among 'train_l', 'train_u' or 'val'")
            
            h5f = h5py.File(self.dir + f"/{flag}_h5/{id}.h5", 'r')
            img = h5f['image'][:]
            
            if self.mode == 'train_l':
                mask = h5f['label'][:]
            elif self.mode == 'train_u':
                mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
            
            h5f.close()
            ignore_value = 254 if self.mode == 'train_u' else 255
            img, mask = crop_3d(img, mask, self.size, ignore_value)
            
            if self.mode == 'train_l':
                return normalize_3d_new(img, mask)             
             
            img_w, img_s = deepcopy(img), deepcopy(img)

            img_s = BrightnessMultiplicativeTransforms(img_s, 0.5, (0.5, 1.5))
            img_s = ContrastAugmentationTransforms(img_s, 0.5, (0.5, 1.5))
            img_s = GammaTransforms(img_s, 0.5, (0.5, 1.5), retain_stats=True, invert_image=False)
            
            cutmix_box = obtain_cutmix_box_3d(img_s)
            
            ignore_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.uint8)
            img_s, ignore_mask = normalize_3d_new(img_s, ignore_mask)
            
            mask = torch.from_numpy(np.array(mask)).long()
            ignore_mask[mask == 254] = 255
            
            return normalize_3d_new(img_w), img_s, ignore_mask, cutmix_box
    
        if self.mode == 'val':
            h5f = h5py.File(self.dir + f"/labeled_h5/{id}.h5", 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            img, mask = normalize_3d_new(img, mask)   
            return img, mask
        
        if self.mode == 'val_test':
            h5f = h5py.File(self.dir + f"/labeled_h5/{id}.h5", 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            ignore_value = 255
            img, mask = crop_3d(img, mask, self.size, ignore_value)
            img, mask = normalize_3d_new(img, mask)
            return img, mask
    
    def __len__(self):
        return len(self.name_list)


class SemiDataset2D(Dataset):
    def __init__(self, mode, cfg=None, size=None, args=None):
        self.mode = mode
        self.cfg = cfg or {}
        self.args = args
        self.root_path = self._get_config_value('root_path', '/data/lhy_data/ACDC')
        self.size = size if size is not None else self._get_config_value('crop_size', None)
        self.images_h5_dir = self._resolve_images_h5_dir()
        self.use_explicit_semi_split = False
        self.name_list = self._build_name_list()
        print(f"{mode} data number is: {len(self.name_list)}")

    def _get_config_value(self, key, default=None):
        if isinstance(self.cfg, dict) and key in self.cfg:
            return self.cfg[key]
        if self.args is not None and hasattr(self.args, key):
            return getattr(self.args, key)
        if isinstance(self.cfg, dict):
            for alt_key in ['base_dir', 'data_root'] if key == 'root_path' else []:
                if alt_key in self.cfg:
                    return self.cfg[alt_key]
        return default

    def _resolve_images_h5_dir(self):
        candidates = ['Images_h5', 'images_h5', 'data/slices', 'data']
        for folder in candidates:
            folder_path = os.path.join(self.root_path, folder)
            if os.path.isdir(folder_path):
                return folder_path
        return os.path.join(self.root_path, 'Images_h5')

    def _resolve_split_file(self):
        if self.mode == 'val_test':
            split_name = 'val.txt'
        elif self.mode == 'test':
            split_name = 'test.txt'
        elif self.mode == 'val':
            split_name = 'val.txt'
        elif self.mode in ['train_l', 'train_u']:
            explicit_path = os.path.join(self.root_path, f'{self.mode}.txt')
            if os.path.exists(explicit_path):
                self.use_explicit_semi_split = True
                return explicit_path
            split_name = 'train.txt'
        else:
            split_name = f'{self.mode}.txt'
        return os.path.join(self.root_path, split_name)

    def _build_name_list(self):
        split_path = self._resolve_split_file()
        with open(split_path, 'r') as f:
            name_list = [item.strip() for item in f.readlines() if item.strip()]

        if self.mode == 'train_l' and not self.use_explicit_semi_split:
            labelnum = self._get_config_value('labelnum', None)
            if labelnum is not None:
                name_list = name_list[:labelnum]
        elif self.mode == 'train_u' and not self.use_explicit_semi_split:
            labelnum = self._get_config_value('labelnum', 0)
            num = self._get_config_value('num', None)
            name_list = name_list[labelnum:]
            if num is not None:
                name_list = name_list[:num]
        return name_list

    def _get_sample_path(self, sample_id):
        sample_name = sample_id if sample_id.endswith('.h5') else f'{sample_id}.h5'
        return os.path.join(self.images_h5_dir, sample_name)

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
        image = torch.from_numpy(image).float().unsqueeze(0)
        return image

    def _to_tensor_mask(self, mask):
        return torch.from_numpy(np.asarray(mask, dtype=np.uint8)).long()

    def _normalize_size(self):
        if self.size is None:
            return None
        if isinstance(self.size, int):
            return self.size
        if isinstance(self.size, (list, tuple)):
            if len(self.size) == 0:
                return None
            if len(self.size) == 1:
                return int(self.size[0])
            if int(self.size[0]) != int(self.size[1]):
                raise ValueError(f"SemiDataset2D expects square crop size, but got {self.size}")
            return int(self.size[0])
        return int(self.size)

    def _apply_train_aug(self, image, mask, ignore_value):
        crop_size = self._normalize_size()
        if crop_size is not None:
            image, mask = crop(image, mask, crop_size, ignore_value=ignore_value)
        image, mask = hflip(image, mask)
        return image, mask

    def __getitem__(self, idx):
        sample_id = self.name_list[idx]
        sample_path = self._get_sample_path(sample_id)
        with h5py.File(sample_path, 'r') as h5f:
            image = h5f['image'][:]
            has_label = 'label' in h5f
            if self.mode == 'train_u':
                mask = np.zeros_like(image, dtype=np.uint8)
            elif has_label:
                mask = h5f['label'][:]
            else:
                raise KeyError(f'label not found in {sample_path}')

        image_pil = Image.fromarray(self._to_uint8_image(image))
        mask_pil = Image.fromarray(np.asarray(mask, dtype=np.uint8))

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

        if self.mode in ['val', 'val_test', 'test']:
            return self._to_tensor_image(image), self._to_tensor_mask(mask)

        raise ValueError(f'Unsupported mode: {self.mode}')

    def __len__(self):
        return len(self.name_list)
