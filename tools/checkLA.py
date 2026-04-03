import os
import argparse
from collections import defaultdict

def get_subfolders(folder_path):
    """获取指定文件夹下的所有子文件夹"""
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    subfolders = {}
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders[item] = item_path
    return subfolders

def get_files_with_sizes(folder_path):
    """获取指定文件夹下的所有文件及其大小"""
    files = {}
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, folder_path)
            files[rel_path] = os.path.getsize(file_path)
    return files

def compare_folders(folder1, folder2):
    """比较两个文件夹的内容"""
    print(f"正在比较文件夹...")
    print(f"文件夹1: {folder1}")
    print(f"文件夹2: {folder2}")
    print("=" * 80)
    
    # 获取子文件夹
    subfolders1 = get_subfolders(folder1)
    subfolders2 = get_subfolders(folder2)
    
    names1 = set(subfolders1.keys())
    names2 = set(subfolders2.keys())
    
    # 检查多余的子文件夹
    only_in_1 = names1 - names2
    only_in_2 = names2 - names1
    
    if only_in_1:
        print("\n❌ 仅在文件夹1中存在的子文件夹:")
        for name in sorted(only_in_1):
            print(f"  - {name}")
    
    if only_in_2:
        print("\n❌ 仅在文件夹2中存在的子文件夹:")
        for name in sorted(only_in_2):
            print(f"  - {name}")
    
    # 检查共同的子文件夹
    common_names = names1 & names2
    if not common_names:
        print("\n❌ 没有共同的子文件夹！")
        return
    
    print(f"\n✅ 找到 {len(common_names)} 个共同的子文件夹，正在比较内容...")
    print("=" * 80)
    
    # 比较每个共同子文件夹的内容
    all_ok = True
    inconsistent_folder_count = 0
    for name in sorted(common_names):
        print(f"\n📁 子文件夹: {name}")
        print("-" * 80)
        
        folder1_sub = subfolders1[name]
        folder2_sub = subfolders2[name]
        
        files1 = get_files_with_sizes(folder1_sub)
        files2 = get_files_with_sizes(folder2_sub)
        
        filenames1 = set(files1.keys())
        filenames2 = set(files2.keys())
        
        # 检查多余的文件
        only_files_in_1 = filenames1 - filenames2
        only_files_in_2 = filenames2 - filenames1
        
        if only_files_in_1:
            all_ok = False
            print(f"  ❌ 仅在文件夹1存在的文件:")
            for f in sorted(only_files_in_1):
                print(f"      - {f}")
        
        if only_files_in_2:
            all_ok = False
            print(f"  ❌ 仅在文件夹2存在的文件:")
            for f in sorted(only_files_in_2):
                print(f"      - {f}")
        
        # 检查共同文件的大小
        common_files = filenames1 & filenames2
        size_mismatch = []
        
        for f in common_files:
            if files1[f] != files2[f]:
                size_mismatch.append((f, files1[f], files2[f]))

        folder_has_issue = bool(only_files_in_1 or only_files_in_2 or size_mismatch)
        if folder_has_issue:
            inconsistent_folder_count += 1
        
        if size_mismatch:
            all_ok = False
            print(f"  ❌ 文件大小不一致:")
            for f, s1, s2 in size_mismatch:
                print(f"      - {f}: 文件夹1={s1} bytes, 文件夹2={s2} bytes")
        
        if not only_files_in_1 and not only_files_in_2 and not size_mismatch:
            print(f"  ✅ 所有文件一致")
    
    print("\n" + "=" * 80)
    print(f"不一致的子文件夹总数: {inconsistent_folder_count}")
    if all_ok:
        print("🎉 所有检查通过！两个文件夹完全一致。")
    else:
        print("⚠️  发现不一致，请检查上述信息。")

def main():
    parser = argparse.ArgumentParser(description="检查两个文件夹及其子文件夹的一致性")
    parser.add_argument("folder1", default="/data/lhy_data/LA_2018_UAMT/2018LA_Seg_Training Set", help="第一个主文件夹路径")
    parser.add_argument("folder2", default="/data/lhy_data/LA2018", help="第二个主文件夹路径")
    
    args = parser.parse_args()
    
    try:
        compare_folders(args.folder1, args.folder2)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
