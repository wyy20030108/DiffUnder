from pathlib import Path

def count_files_pathlib(folder_path):
    path = Path(folder_path)
    
    # 统计当前目录下（不递归）
    # count = len([f for f in path.iterdir() if f.is_file()])
    
    # 递归统计所有子目录
    count = len([f for f in path.rglob('*') if f.is_file()])
    
    return count

path = "/data/lhy_data/ACDC/Masks"
print(f"文件总数: {count_files_pathlib(path)}")


# ACDC: Mask:300 Images:300