import os
import glob
import random

def generate_busi_base_splits(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    print("🔄 开始扫描并统计 BUSI 数据集...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 获取所有的图片文件
    all_files = glob.glob(os.path.join(data_dir, "*.png"))

    unique_cases = []
    for f in all_files:
        filename = os.path.basename(f)
        
        # 核心过滤逻辑：凡是名字里带 "_mask" 的统统跳过
        # 这样不管它是 _mask.png 还是 _mask_1.png，都能被过滤掉
        # 最终只留下 "benign (1).png" 这种纯净的原图名字
        if "_mask" not in filename:
            # 去掉后缀，得到基础名字，例如 "benign (1)"
            case_name = filename.replace(".png", "")
            unique_cases.append(case_name)

    total_num = len(unique_cases)
    print(f"📄 统计完成：共找到 {total_num} 个有效的数据样本 (Cases)。")

    # 2. 固定随机种子并打乱
    # 保证每次跑出来的 txt 列表顺序永远一致
    random.seed(2026)
    unique_cases.sort()  # 先排序，确保不同操作系统的文件系统读取顺序不会影响最终结果
    random.shuffle(unique_cases)

    # 3. 按比例计算数量 (默认 80% Train, 10% Val, 10% Test)
    train_num = int(total_num * train_ratio)
    val_num = int(total_num * val_ratio)
    test_num = total_num - train_num - val_num

    # 4. 执行划分
    train_cases = unique_cases[:train_num]
    val_cases = unique_cases[train_num:train_num + val_num]
    test_cases = unique_cases[train_num + val_num:]

    print(f"✅ 划分结果: Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_cases)}")

    # 5. 写入 txt 文件
    def write_txt(cases, filename):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(cases) + '\n')
        print(f"   -> 已保存至 {filename}")

    write_txt(train_cases, 'train.txt')
    write_txt(val_cases, 'val.txt')
    write_txt(test_cases, 'test.txt')

    print("\n💡 终极提示：")
    print("   你的 DataLoader 现在只需要加载这三个 txt 文件。")
    print("   对于半监督训练，直接读取 train_slice.txt 得到一个 list，然后用 list[:num] 作为有标签，list[num:] 作为无标签即可！")

# ==========================================
# 运行配置区域
# ==========================================
if __name__ == "__main__":
    # 填入你截图里那个把所有 benign 和 malignant 图片混在一起的目录路径
    DATA_DIR = "/data/lhy_data/BUSI/Dataset_BUSI_with_GT/without_normal" 
    
    # 填入生成的 txt 文件想要保存的路径
    OUTPUT_DIR = "/data/lhy_data/BUSI/Dataset_BUSI_with_GT/busi_splits" 
    
    generate_busi_base_splits(DATA_DIR, OUTPUT_DIR)