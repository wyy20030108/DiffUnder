import os
import glob

def check_dataset_match(txt_path, h5_dir):
    """
    检查生成的 H5 文件是否与指定的 TXT 划分列表匹配
    """
    print("🔍 开始检查数据集匹配情况...\n")

    # 1. 读取 TXT 文件中的切片名称
    if not os.path.exists(txt_path):
        print(f"❌ 找不到 TXT 文件: {txt_path}")
        return
        
    with open(txt_path, 'r') as f:
        # 去除换行符和空格，并过滤掉空行
        txt_slices = set([line.strip() for line in f.readlines() if line.strip()])

    # 2. 获取 H5 文件夹中的切片名称
    if not os.path.exists(h5_dir):
        print(f"❌ 找不到 H5 文件夹: {h5_dir}")
        return
        
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))
    # 提取纯文件名（去掉路径和 .h5 后缀），以匹配 txt 中的格式
    h5_slices = set([os.path.basename(f).replace('.h5', '') for f in h5_files])

    # 3. 输出基础数量信息
    print(f"📄 TXT 文件中包含的切片数量: {len(txt_slices)}")
    print(f"📁 H5 文件夹中实际的切片数量: {len(h5_slices)}\n")

    # 4. 进行集合比对 (Set Operations)
    missing_in_h5 = txt_slices - h5_slices  # 在 txt 中，但在 h5 文件夹中找不到
    extra_in_h5 = h5_slices - txt_slices    # 在 h5 文件夹中，但 txt 中没有要求

    # 5. 判断结果
    is_strict_equal = (len(missing_in_h5) == 0 and len(extra_in_h5) == 0)

    if is_strict_equal:
        print("✅ 完美匹配！你的 H5 数据与 TXT 文件完全一致（不多也不少）。")
    else:
        print("⚠️ 数量不完全匹配，具体诊断如下：")
        
        # 致命错误：缺少文件
        if len(missing_in_h5) > 0:
            print(f"   ❌ [致命] 你的 H5 文件夹中缺少了 {len(missing_in_h5)} 个 TXT 里要求的切片！")
            print(f"      缺失示例: {list(missing_in_h5)[:5]} ...")
            print("      结论：这个 TXT 列表你目前【不能用】，因为你缺少运行它所需的数据。")
        else:
            print("   ✅ [良好] TXT 列表中要求的所有切片，在你的 H5 文件夹中都能找到（没有缺失）。")

        # 正常现象：有多余文件
        if len(extra_in_h5) > 0:
            print(f"   ℹ️ [提示] 你的 H5 文件夹中多出了 {len(extra_in_h5)} 个 TXT 里没有的切片。")
            if len(missing_in_h5) == 0:
                print("\n💡 【最终结论】：这个 TXT 文件你【完全可以使用】！")
                print("   说明：H5 文件夹比 TXT 多出文件是正常的（因为 H5 往往是全集，而 TXT 只是训练集划分）。")
                print("   DataLoader 在读取时，只会去加载 TXT 里列出的文件，多余的 H5 文件会安全地待在文件夹里，不会影响训练。")

# ==========================================
# 运行配置区域
# ==========================================
if __name__ == "__main__":
    # 替换为你实际的路径
    TXT_FILE_PATH = "/data/lhy_data/ACDC/train_slice.txt"
    H5_FOLDER_PATH = "/data/lhy_data/ACDC/Images_h5"  # 填入你上一步生成的 2978 个 h5 文件的目录

    check_dataset_match(TXT_FILE_PATH, H5_FOLDER_PATH)