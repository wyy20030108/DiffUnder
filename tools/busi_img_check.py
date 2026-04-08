import os
import glob
import cv2
from collections import Counter

def check_png_shapes_and_mins(data_dir):
    print(f"🔍 正在扫描目录: {data_dir}")
    
    # 递归获取目录下所有的 .png 文件
    png_files = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    
    if not png_files:
        print("❌ 未找到任何 .png 文件，请检查路径。")
        return
        
    print(f"📄 共找到 {len(png_files)} 个 .png 文件。正在极速读取尺寸...\n")
    
    shape_counter = Counter()
    error_files = []
    
    # 初始化前三个维度的最小值为正无穷大
    min_h = float('inf')
    min_w = float('inf')
    min_c = float('inf')
    
    for file_path in png_files:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            error_files.append(file_path)
            continue
            
        shape = img.shape
        shape_counter[shape] += 1
        
        # 提取 H, W, C
        h = shape[0]
        w = shape[1]
        # 如果是单通道灰度图，shape 只有两个元素 (H, W)，此时通道数视为 1
        c = shape[2] if len(shape) > 2 else 1
        
        # 动态更新全局最小值
        if h < min_h: min_h = h
        if w < min_w: min_w = w
        if c < min_c: min_c = c
        
    # 打印最终的汇总结果
    print("-" * 50)
    print("📊 数据集 Shape 统计结果 (高度 H, 宽度 W, 通道数 C):")
    print("-" * 50)
    for shape, count in shape_counter.most_common():
        print(f"   👉 Shape {shape}: 共有 {count} 张图片")
        
    print("-" * 50)
    print("📏 前三个维度的全局最小值:")
    print("-" * 50)
    # 如果没找到任何有效图片，这里可能还是 inf，做个容错保护
    if min_h == float('inf'):
        print("   ❌ 未能读取到有效的图片尺寸。")
    else:
        print(f"   - 最小高度 (Min H): {min_h} 像素")
        print(f"   - 最小宽度 (Min W): {min_w} 像素")
        print(f"   - 最小通道数 (Min C): {min_c}")
    print("-" * 50)
    
    if error_files:
        print(f"\n⚠️ 警告：有 {len(error_files)} 个文件损坏或无法读取！")
        for f in error_files[:5]: 
            print(f"      - {os.path.basename(f)}")

    # 给出针对深度学习的专业建议
    if min_h != float('inf') and min_w != float('inf'):
        print("\n💡 终极提示：")
        print(f"   如果要进行 RandomCrop 或 CenterCrop，你的 crop_size 绝对不能超过 ({min_h}, {min_w})！")
        print("   如果必须设得更大，请记得在 DataLoader 中开启 Padding (如 cv2.BORDER_CONSTANT 补黑边)。")

# ==========================================
# 运行配置区域
# ==========================================
if __name__ == "__main__":
    # 填入你要检查的文件夹路径 (支持穿透子文件夹)
    TARGET_DIR = r"/data/lhy_data/BUSI/Dataset_BUSI_with_GT/without_normal" 
    
    check_png_shapes_and_mins(TARGET_DIR)