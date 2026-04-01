import itertools, torch
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import h5py
from copy import deepcopy

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