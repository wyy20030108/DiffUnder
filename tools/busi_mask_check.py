import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def check_mask_unique_values(dataset_dir):
    """
    遍历指定目录下所有文件名包含 'mask' 的 PNG 图片，检查并统计它们的 unique 像素值。
    """
    dataset_path = Path(dataset_dir)
    
    # 递归查找所有文件名中包含 'mask' 的 .png 文件
    # 如果你的数据没有嵌套在子文件夹中，也可以用 glob("*mask*.png")
    mask_files = list(dataset_path.rglob("*mask*.png"))
    
    if not mask_files:
        print("未找到任何 mask 文件，请检查路径是否正确！")
        return

    print(f"共找到 {len(mask_files)} 个 Mask 文件。开始检查...\n")

    # 用于记录全局出现过的所有 unique 值
    global_unique_values = set()
    # 用于统计不同 unique 值组合出现的次数 (例如有多少张图只有 [0]，有多少张图有 [0, 255])
    value_combinations_count = defaultdict(int)
    # 记录出现异常值的具体文件路径
    anomalous_files = []

    for mask_path in mask_files:
        # 以灰度模式读取图像
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"警告: 无法读取文件 {mask_path.name}")
            continue

        # 获取当前图像的所有 unique 像素值
        unique_vals = np.unique(mask)
        
        # 将局部 unique 值加入全局集合
        global_unique_values.update(unique_vals)
        
        # 记录这种值组合出现的次数（将 numpy array 转为 tuple 以便作为字典的 key）
        val_tuple = tuple(unique_vals)
        value_combinations_count[val_tuple] += 1

        # 异常值检测：如果包含除了 0 和 255 之外的值，记录下来
        if not set(unique_vals).issubset({0, 255}):
            anomalous_files.append((mask_path.name, unique_vals))

    # --- 打印统计结果 ---
    print("-" * 40)
    print("【全局唯一像素值 (Global Unique Values)】:")
    print(sorted(list(global_unique_values)))
    
    print("\n【像素值组合统计 (Value Combinations Count)】:")
    for vals, count in value_combinations_count.items():
        print(f"  包含像素值 {list(vals)} 的 Mask 数量: {count} 张")

    print("\n【异常值检查 (Anomaly Check)】:")
    if not anomalous_files:
        print("  ✅ 恭喜！所有 Mask 均非常干净，只包含 0 或 255（或全 0）。")
    else:
        print(f"  ⚠️ 警告！发现 {len(anomalous_files)} 个 Mask 包含异常值（非 0 且非 255）：")
        # 打印前 5 个异常文件作为示例
        for name, vals in anomalous_files[:5]:
            print(f"    - {name}: {vals}")
        if len(anomalous_files) > 5:
            print("    - ... (省略更多)")

# ==========================================
# 使用方法：将下面的路径替换为你存放图片文件夹的实际路径
# 例如：'./Dataset_BUSI_with_GT/benign' 或者包含所有类别的父目录
# ==========================================
if __name__ == "__main__":
    dataset_directory = "/data/lhy_data/BUSI/Dataset_BUSI_with_GT/without_normal" 
    check_mask_unique_values(dataset_directory)