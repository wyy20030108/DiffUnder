import os
import glob
import re
from collections import defaultdict

def ultimate_fix_acdc_names(h5_dir):
    print("🚀 开始执行终极修复：动态映射 Frame 并执行 Slice +1 ...")
    
    h5_files = glob.glob(os.path.join(h5_dir, "patient*.h5"))
    if not h5_files:
        print("❌ 在指定目录没有找到 H5 文件！")
        return

    # 1. 将文件按 patient_id 分组
    patient_data = defaultdict(list)
    for file_path in h5_files:
        filename = os.path.basename(file_path)
        match = re.match(r"(patient\d+)_frame(\d+)_slice_(\d+)\.h5", filename)
        if match:
            patient_data[match.group(1)].append({
                'original_path': file_path,
                'frame': match.group(2),
                'slice': int(match.group(3))
            })

    if not patient_data:
        print("❌ 没有找到符合 patientXXX_frameYY_slice_Z.h5 的文件！")
        return

    rename_tasks = []

    # 2. 为每个病人动态分配 Frame 号
    for pid, files in patient_data.items():
        # 提取该病人拥有的所有不重复的 frame 号，并按数字大小排序
        unique_frames = sorted(list(set([f['frame'] for f in files])), key=int)
        
        # 创建映射字典 (较小的帧 -> 01, 较大的帧 -> 02)
        frame_map = {}
        if len(unique_frames) >= 1:
            frame_map[unique_frames[0]] = "01"
        if len(unique_frames) >= 2:
            frame_map[unique_frames[1]] = "02"
            
        # 生成该病人的重命名任务
        for info in files:
            new_frame = frame_map.get(info['frame'], "01") # 获取映射后的帧号
            new_slice = info['slice'] + 1                  # slice 索引加 1
            
            new_filename = f"{pid}_frame{new_frame}_slice_{new_slice}.h5"
            new_path = os.path.join(h5_dir, new_filename)
            
            rename_tasks.append((info['original_path'], new_path))

    # 3. 安全执行重命名 (两步走防覆盖机制)
    print(f"📦 正在处理 {len(rename_tasks)} 个文件...")
    
    # 第一步：全部加上 .tmp 后缀，避免互相覆盖
    for old_path, _ in rename_tasks:
        os.rename(old_path, old_path + ".tmp")
        
    # 第二步：去掉 .tmp 并改为最终目标名字
    rename_count = 0
    for old_path, new_path in rename_tasks:
        os.rename(old_path + ".tmp", new_path)
        rename_count += 1

    print(f"✅ 完美修复！总共成功重命名了 {rename_count} 个 H5 文件。")
    print("💡 解析结果示例：")
    print("   - 正常的 patient062: frame01 -> 01, frame09 -> 02")
    print("   - 特殊的 patient090: frame04 -> 01, frame11 -> 02")
    print("   - 所有的 slice_0 都变成了 slice_1")
    print("➡️ 你现在可以放心地运行 txt 匹配检查脚本了！")

# ==========================================
# 运行配置区域
# ==========================================
if __name__ == "__main__":
    H5_FOLDER_PATH = "/data/lhy_data/ACDC/Images_h5"  # 你的 H5 文件夹路径
    ultimate_fix_acdc_names(H5_FOLDER_PATH)