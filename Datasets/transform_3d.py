import numpy as np
import random
import torch
import cv2
from PIL import Image
from scipy.ndimage import zoom, rotate
from torchvision import transforms
from typing import Tuple, Union, Callable


def flip_3d(img, mask):
    """img, mask : 3d numpy array or tensor
    """
    p = np.random.randint(3)
    if p == 0:
        img_f, mask_f = img[::-1, :, :], mask[::-1, :, :]
    elif p == 1:
        img_f, mask_f = img[:, ::-1, :], mask[:, ::-1, :]
    elif p == 2:
        img_f, mask_f = img[:, :, ::-1], mask[:, :, ::-1]
    
    return img_f, mask_f

def extract_minority_mask_crop(label_crop, num_classes_list, crop_size):
    minority_mask = np.isin(label_crop, num_classes_list)
    coords = np.where(minority_mask)
    if coords[0].size == 0:
        # 如果没有少数类，则在整个图像范围内进行随机裁切作为回退策略
        d, h, w = label_crop.shape
        min_crop1 = np.random.randint(0, d - crop_size[0] + 1)
        min_crop2 = np.random.randint(0, h - crop_size[1] + 1)
        min_crop3 = np.random.randint(0, w - crop_size[2] + 1)
        
        max_crop1 = min_crop1 + crop_size[0]
        max_crop2 = min_crop2 + crop_size[1]
        max_crop3 = min_crop3 + crop_size[2]
        
        return (slice(min_crop1, max_crop1), slice(min_crop2, max_crop2), slice(min_crop3, max_crop3))
    
    min_bottom1, max_bottom1 = label_crop.shape[0], 0
    min_bottom2, max_bottom2 = label_crop.shape[1], 0
    min_bottom3, max_bottom3 = label_crop.shape[2], 0
    for cls in num_classes_list:
        if cls not in np.unique(label_crop).tolist():
            continue
        # 这里可以添加提取少数类掩码的逻辑
        cls_mask1 = np.unique(np.where(label_crop==cls)[0])
        cls_mask2 = np.unique(np.where(label_crop==cls)[1])
        cls_mask3 = np.unique(np.where(label_crop==cls)[2])
        min_bottom1, max_bottom1 = min(cls_mask1.min(), min_bottom1), max(cls_mask1.max(), max_bottom1)
        min_bottom2, max_bottom2 = min(cls_mask2.min(), min_bottom2), max(cls_mask2.max(), max_bottom2)
        min_bottom3, max_bottom3 = min(cls_mask3.min(), min_bottom3), max(cls_mask3.max(), max_bottom3)
    if (max_bottom1 - min_bottom1) >= crop_size[0]:
        min_b1 = max_bottom1 - crop_size[0]
        min_crop1 = np.random.randint(min_bottom1, min_b1 + 1)
        if min_crop1 < 0:
            min_crop1 = 0
        max_crop1 = min_crop1 + crop_size[0]
    else:
        min_crop1 = np.random.randint(max(0, min_bottom1 - int(crop_size[0] / 2)), min_bottom1 + 1)
        max_crop1 = min_crop1 + crop_size[0]

    if (max_bottom2 - min_bottom2) >= crop_size[1]:
        min_b2 = max_bottom2 - crop_size[1]
        min_crop2 = np.random.randint(min_bottom2, min_b2 + 1)
        if min_crop2 < 0:
            min_crop2 = 0
        max_crop2 = min_crop2 + crop_size[1]
    else:
        min_crop2 = np.random.randint(max(0, min_bottom2 - int(crop_size[1] / 2)), min_bottom2 + 1)
        max_crop2 = min_crop2 + crop_size[1]

    if (max_bottom3 - min_bottom3) >= crop_size[2]:
        min_b3 = max_bottom3 - crop_size[2]
        min_crop3 = np.random.randint(min_bottom3, min_b3 + 1)
        if min_crop3 < 0:
            min_crop3 = 0
        max_crop3 = min_crop3 + crop_size[2]
    else:
        min_crop3 = np.random.randint(max(0, min_bottom3 - int(crop_size[2] / 2)), min_bottom3 + 1)
        max_crop3 = min_crop3 + crop_size[2]
    mask_crop = (slice(min_crop1, max_crop1, None), slice(min_crop2, max_crop2, None), slice(min_crop3, max_crop3, None))
    
    return  mask_crop

def extract_minority_mask_crop_new(label_crop, num_classes_list, crop_size):
    minority_mask = np.isin(label_crop, num_classes_list)
    coords = np.where(minority_mask)
    if coords[0].size == 0:
        # 如果没有少数类，则在整个图像范围内进行随机裁切作为回退策略
        d, h, w = label_crop.shape
        min_crop1 = np.random.randint(0, d - crop_size[0] + 1)
        min_crop2 = np.random.randint(0, h - crop_size[1] + 1)
        min_crop3 = np.random.randint(0, w - crop_size[2] + 1)
        max_crop1 = min_crop1 + crop_size[0]
        max_crop2 = min_crop2 + crop_size[1]
        max_crop3 = min_crop3 + crop_size[2]
        return (slice(min_crop1, max_crop1), slice(min_crop2, max_crop2), slice(min_crop3, max_crop3))
    
    min_bounds = np.array([ax.min() for ax in coords])
    max_bounds = np.array([ax.max() for ax in coords])
    crop_starts = []
    for i in range(3):
        min_b, max_b = min_bounds[i], max_bounds[i]
        span = max_b - min_b
        
        if span >= crop_size[i]: # 边界框大于等于裁切尺寸，在有效范围内随机选择起点
            start_range_min = min_b
            start_range_max = max_b - crop_size[i]
            start_point = np.random.randint(start_range_min, start_range_max + 1)
        else: # 边界框小于裁切尺寸，在以边界框为中心的一定范围内选择起点 # 确保起点不会导致裁切超出图像范围
            start_range_min = max(0, min_b - int((crop_size[i] - span) / 2))
            start_range_max = min_b
            # 防止 起点+crop_size 超出图像边界
            if start_range_min + crop_size[i] > label_crop.shape[i]:
                start_range_min = label_crop.shape[i] - crop_size[i]
            if start_range_max + crop_size[i] > label_crop.shape[i]:
                start_range_max = label_crop.shape[i] - crop_size[i]
            start_point = np.random.randint(start_range_min, start_range_max + 1)
        crop_starts.append(start_point)
    crop_starts = np.array(crop_starts)
    crop_ends = crop_starts + np.array(crop_size)
    mask_crop = tuple(slice(start, end) for start, end in zip(crop_starts, crop_ends))
    return  mask_crop

def crop_3d(img, mask, size, ignore_value=255, crop_m=False, mgnet=False, gnet=False, minority=None, mask_crop=None):
    """对三维图像进行随机裁剪image, mask (numpy.ndarray), ignore_value (int): 填充区域的掩码值.
    """
    if mask_crop is None:
        h, l, w= img.shape
        # u_size = (100, 330, 330)
        # l_size = (119, 270, 316)
        # if mode == 'train_l':
        #     size = (119, 270, 316)
        # elif mode == 'train_u':
        #     size = (100, 330, 330)
        while mask.shape[0] <= size[0] or mask.shape[1] <= size[1] or mask.shape[2] <= size[2]:
            ph = int(max((size[0] - h) // 2 + 1, 0))
            pl = int(max((size[1] - l) // 2 + 1, 0))
            pw = int(max((size[2] - w) // 2 + 1, 0))
            img = np.pad(img, [(ph, ph), (pl,pl), (pw, pw)],
                        mode='constant', constant_values=0)
            mask = np.pad(mask, [(ph, ph), (pl,pl), (pw, pw)],
                        mode='constant', constant_values=ignore_value)

        h, l, w = img.shape
        h1 = np.random.randint(0, h - size[0])
        l1 = np.random.randint(0, l - size[1])
        w1 = np.random.randint(0, w - size[2])
        
        # 处理溢出值, 处理为size大小
        mask_crop = np.s_[h1:h1 + size[0], l1:l1 + size[1], w1:w1 + size[2]]
        label = mask[mask_crop]
        img = img[mask_crop]
        if crop_m:
            return img, label, mask_crop
        return img, label
    else:
        img = img[mask_crop]
        label = mask[mask_crop]
        return img, label, mask_crop
        
        
def resize_3d(img, mask, ratio_range, n_class):
    ratio = random.uniform(ratio_range[0], ratio_range[1])
    img = zoom(img, (ratio , ratio , ratio), mode='nearest')
    mask = zoom(mask, (ratio , ratio , ratio), mode='nearest')
    mask = mask.clip(0, n_class - 1)
    return img, mask


# scipy.ndimage.gaussian_filter
def blur_3d(img):
    sigma = np.random.randint(2, 10)
    if sigma % 2 == 0:
        sigma += 1
    img = cv2.GaussianBlur(img, (sigma, sigma), 0, 0)
    return img


def random_rotate_3d(img, ax_range):
    # angle = np.random.uniform(0, 360.0)
    angle = np.random.uniform(0, 90.0)
    img_rotate = rotate(img, angle, axes=ax_range, reshape=True, order=3, mode='grid-constant', cval=0)
    return img_rotate


def gamma_transform_3d(img, gamma):
    out = np.array(np.power((img/255), gamma)*255, dtype=np.uint8)
    adjusted = np.clip(out, 0, 255)
    return adjusted


# 对于医学图像来说可有可无
def normalize_3d(img, mask=None):
    # mean_val, std_val = 0.03775, 0.25098
    img = np.array(img, dtype=np.float32)
    # img = (img - mean_val) / std_val
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = torch.from_numpy(np.array(img)).float()

    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def normalize_3d_new(img, mask=None):
    img = np.array(img, dtype=np.float32)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = torch.from_numpy(np.array(img)).float()

    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

# 颜色增强， img: numpy array
def ColorJitter3d(image, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
    max = image.shape[0]
    # img_3d = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.uint8)
    img_3d_list = []
    color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    # 从l维度开始施加colorjitter
    for i in range(max):
        img = image[i]
        if np.max(img) - np.min(img) != 0:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = Image.fromarray(np.uint8(255*img))
        else:
            img = (img - np.min(img))
            img = Image.fromarray(np.uint8(255*img))
        img = color_jitter(img)
        img = np.array(img)
        img = np.reshape(img, (img.shape[0], img.shape[1]))
        
        # if len(np.unique(img_3d[0])) == 1 and int(np.unique(img_3d[0])) == 0:
        #     img_3d = img
        # else:
        #     img_3d = np.concatenate((img_3d, img), axis=0)
        img_3d_list.append(img)  
        
    img_3d = np.stack(img_3d_list, axis=0)
    img_3d = img_3d.clip(0, 255)
            
    return img_3d

# cutmix_3d
def obtain_cutmix_box_3d(img, size_min=0.01, size_max=0.5,ratio_min = 0.5):
    l, w, h = img.shape
    mask = torch.zeros(l, w, h)
    ratio_max = 1 / ratio_min
    size = np.random.uniform(size_min, size_max) * min(l, w, h) * max(l, w, h)
    
    while True:
        ratio = np.random.uniform(ratio_min, ratio_max)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        cutmix_l = np.random.randint(0, l)
        cutmix_l = np.random.randint(cutmix_l, l)
        x = np.random.randint(0, cutmix_w)
        y = np.random.randint(0, cutmix_h)
        # z = np.random.randint(0, cutmix_l)
        if cutmix_l > 0:
            z = np.random.randint(0, cutmix_l)
            z = np.random.randint(z, cutmix_l)
        else:
            z = 0
        x = np.random.randint(x, cutmix_w)
        y = np.random.randint(y, cutmix_h)
        # z = np.random.randint(z, cutmix_l)

        if z + cutmix_l <= l and x + cutmix_w <= w and y + cutmix_h <= h:
            break
    mask[z:z + cutmix_l, x:x + cutmix_w, y:y + cutmix_h] = 1
    
    return mask

# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def BrightnessMultiplicativeTransforms(data_ori: np.ndarray, p_per_sample=0.5, multiplier_range=(0.7, 1.3)) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray): (D, H, W), don't has channel and batch dim.
        multiplier_range (tuple, optional): _description_. Defaults to (0.7, 1.3).
        p_per_sample (float, optional): _description_. Defaults to 0.3.
    """
    data = data_ori.copy()
    d, h, w = data.shape
    for b in range(0, d):
        if np.random.uniform() < p_per_sample:
            for c in range(0, h):
                multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
                data[b][c] *= multiplier
    return data

def ContrastAugmentationTransforms(data_ori: np.ndarray, p_per_sample: float = 0.5, 
                                       contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),) -> np.ndarray:
    data = data_ori.copy()
    d, h, w = data.shape
    for b in range(0, d):
        for c in range(0, h):
            if np.random.uniform() < p_per_sample:
                if callable(contrast_range):
                    factor = contrast_range()
                elif np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor = np.random.uniform(contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
                mn = data[b][c].mean()
                minm = data[b][c].min()
                maxm = data[b][c].max()

                data[b][c] = (data[b][c] - mn) * factor + mn
                data[b][c][data[b][c] < minm] = minm
                data[b][c][data[b][c] > maxm] = maxm
    return data

def GammaTransforms(data_ori: np.ndarray, p_per_sample: float = 0.5,
                        gamma_range=(0.5, 2),
                        retain_stats: Union[bool, Callable[[], bool]] = False, epsilon=1e-15, invert_image=False
                        ) -> np.ndarray:
    data = data_ori.copy()
    if invert_image:
        data = - data

    d, h, w = data.shape
    for b in range(0, d):
        for c in range(0, h):
            if np.random.uniform() < p_per_sample:
                retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
                if retain_stats_here:
                    mn = data[b][c].mean()
                    sd = data[b][c].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data[b][c].min()
                rnge = data[b][c].max() - minm
                data[b][c] = np.power(((data[b][c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                if retain_stats_here:
                    data[b][c] = data[b][c] - data[b][c].mean()
                    data[b][c] = data[b][c] / (data[b][c].std() + 1e-8) * sd
                    data[b][c] = data[b][c] + mn
    
    if invert_image:
        data = - data
    return data






