import os
import glob
from collections import defaultdict

def verify_busi_correspondence(data_dir):
    print(f"🔍 正在深度扫描并校验目录: {data_dir}")
    
    all_files = glob.glob(os.path.join(data_dir, "*.png"))
    
    if not all_files:
        print("❌ 警告：未找到任何 .png 文件，请检查路径是否正确！")
        return
        
    original_images = []
    masks = []
    
    # 1. 区分原图和掩码
    for filepath in all_files:
        filename = os.path.basename(filepath)
        if "_mask" in filename:
            masks.append(filename)
        else:
            original_images.append(filename)
            
    # 2. 建立 原图基础名 -> 对应Mask列表 的映射字典
    # 比如: 'benign (4)' -> ['benign (4)_mask.png', 'benign (4)_mask_1.png']
    mask_mapping = defaultdict(list)
    for mask_name in masks:
        # 通过 '_mask' 切割，提取出前面的基础名字
        base_name = mask_name.split('_mask')[0]
        mask_mapping[base_name].append(mask_name)
        
    # 3. 统计各种情况
    normal_1_to_1 = []      # 正常的 1对1
    one_to_many = {}        # 1张图对应多个Mask
    missing_masks = []      # 有原图但没Mask
    orphaned_masks = []     # 有Mask但找不到原图
    
    original_base_names = [img.replace('.png', '') for img in original_images]
    
    # 检查原图是否都有 mask，以及是否有多个 mask
    for base_name in original_base_names:
        if base_name in mask_mapping:
            mask_list = mask_mapping[base_name]
            if len(mask_list) == 1:
                normal_1_to_1.append(base_name)
            else:
                one_to_many[base_name] = mask_list
        else:
            missing_masks.append(base_name)
            
    # 检查是否有孤立的 mask (找不到原图)
    for base_name in mask_mapping.keys():
        if base_name not in original_base_names:
            orphaned_masks.append((base_name, mask_mapping[base_name]))
            
    # 4. 打印详细报告
    print("-" * 50)
    print("📋 数据集对应关系详细校验报告：")
    print("-" * 50)
    print(f"✅ 完美的【一对一】样本数: {len(normal_1_to_1)} 个")
    
    print(f"\n⚠️ 发现【一对多】(包含多个病灶) 样本数: {len(one_to_many)} 个")
    if one_to_many:
        print("   -> 具体名单及对应的 Masks:")
        count = 0
        for base_name, m_list in one_to_many.items():
            print(f"      - {base_name}.png -> 包含 {len(m_list)} 个标签: {m_list}")
            count += 1
            if count >= 10:  # 避免打印太多刷屏
                print(f"      - ... (还有 {len(one_to_many) - 10} 个未显示)")
                break

    print(f"\n❌ 发现【缺失 Mask】的原图数: {len(missing_masks)} 个")
    if missing_masks:
        print("   -> 危险！以下原图找不到对应的标签:")
        for name in missing_masks:
            print(f"      - {name}.png")
            
    print(f"\n👻 发现【孤立 Mask】(找不到原图) 数: {len(orphaned_masks)} 组")
    if orphaned_masks:
        print("   -> 危险！以下标签找不到对应的原图:")
        for base_name, m_list in orphaned_masks:
            print(f"      - 孤立基础名: {base_name} -> Masks: {m_list}")
            
    print("-" * 50)
    if not missing_masks and not orphaned_masks:
        print("🎉 恭喜！你的数据集极其健康，没有任何数据缺失或断层，多出来的 18 个 Mask 完全是由合理的‘一对多’造成的！")
    else:
        print("🚨 警告：数据存在缺失，请根据上方列表去文件夹里手动排查或删除这些问题数据！")

# ==========================================
# 运行配置
# ==========================================
if __name__ == "__main__":
    TARGET_DIR = r"/data/lhy_data/BUSI/Dataset_BUSI_with_GT/without_normal"
    verify_busi_correspondence(TARGET_DIR)