
# ---------------------------------------------------------------
# This file is adapted from:
# https://github.com/HiLab-git/SSL4MIS/blob/master/code/dataloaders/acdc_data_processing.py
#
# Original Project: SSL4MIS (Semi-Supervised Learning for Medical Image Segmentation)
# Author: HiLab-git
# ---------------------------------------------------------------

import glob
import os
import h5py
import numpy as np
import SimpleITK as sitk
sitk.ProcessObject_GlobalWarningDisplayOff()

def preprocess_nii_to_h5(image_dir, label_dir, output_dir):
    """
    将 3D NIfTI 图像及标签转换为 2D H5 切片。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    slice_num = 0

    # 获取所有 nii.gz 图像路径
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    
    if not image_paths:
        print(f"在 {image_dir} 中没有找到 .nii.gz 文件！")
        return

    for img_path in image_paths:
        # 1. 读取图像
        img_itk = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(img_itk)

        # 2. 定位并读取对应的标签
        # 【修改这里】：根据你的标签命名规则调整。
        # 假设你的图像叫 "case_01.nii.gz"，标签叫 "case_01_gt.nii.gz"
        filename = os.path.basename(img_path)
        label_name = filename.replace(".nii.gz", "_gt.nii.gz") 
        msk_path = os.path.join(label_dir, label_name)

        if not os.path.exists(msk_path):
            print(f"跳过：找不到对应的标签文件 {msk_path}")
            continue

        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)

        # 3. 形状检查
        if image.shape != mask.shape:
            print(f"错误：{filename} 的图像和标签形状不匹配！图像:{image.shape}, 标签:{mask.shape}")
            continue

        # 4. 图像归一化处理
        # 【修改这里】：如果是 CT 数据，建议在此处添加 HU 值截断逻辑
        # 例如软组织窗： image = np.clip(image, -100, 250) 
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) # 加上 1e-8 防止除以 0
        image = image.astype(np.float32)

        item_name = filename.split(".")[0]

        # 5. 3D 转 2D 切片并保存
        for slice_ind in range(image.shape[0]):
            # 可以选择过滤掉全背景（没有器官标签）的切片以加速训练
            # if np.sum(mask[slice_ind]) == 0:
            #     continue
            
            h5_path = os.path.join(output_dir, f"{item_name}_slice_{slice_ind}.h5")
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('image', data=image[slice_ind], compression="gzip")
                f.create_dataset('label', data=mask[slice_ind], compression="gzip")
            slice_num += 1

        print(f"已处理: {item_name}, 生成切片数: {image.shape[0]}")

    print(f"预处理完成！总计生成了 {slice_num} 张二维切片。")

# ==========================================
# 运行脚本
# ==========================================
if __name__ == "__main__":
    # 在这里填入你的实际本地路径
    IMAGE_DIR = "/data/lhy_data/ACDC/Images"
    LABEL_DIR = "/data/lhy_data/ACDC/Masks"
    OUTPUT_DIR = "/data/lhy_data/ACDC/Images_h5"

    preprocess_nii_to_h5(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR)