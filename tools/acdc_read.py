import os
import h5py
from collections import Counter

# 你的 H5 文件夹路径
H5_FOLDER_PATH = "/data/lhy_data/ACDC/Images_h5"

def check_h5_shapes(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ 路径不存在: {folder_path}")
        return
        
    # 获取所有 .h5 文件
    h5_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])
    
    if not h5_files:
        print(f"⚠️ 在 {folder_path} 中没有找到 .h5 文件。")
        return
        
    print(f"🔍 共找到 {len(h5_files)} 个 .h5 文件，开始检查 shape...\n")
    
    # 用于统计 shape 分布
    image_shapes = Counter()
    label_shapes = Counter()
    
    for file_name in h5_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with h5py.File(file_path, 'r') as h5f:
                # 检查文件中是否包含 image 和 label
                if 'image' in h5f and 'label' in h5f:
                    img_shape = h5f['image'].shape
                    lbl_shape = h5f['label'].shape
                    
                    # 打印每个文件的详细 shape（如果嫌终端输出太多，可以注释掉下面这行）
                    print(f"文件: {file_name:<25} | Image: {str(img_shape):<15} | Label: {str(lbl_shape)}")
                    
                    # 记录 shape 次数
                    image_shapes[img_shape] += 1
                    label_shapes[lbl_shape] += 1
                else:
                    print(f"⚠️ 文件 {file_name} 缺失 'image' 或 'label' 键。")
        except Exception as e:
            print(f"❌ 读取文件 {file_name} 时发生错误: {e}")
            
    # 打印最终的统计结果
    print("\n" + "="*40)
    print("📈 Shape 统计总结")
    print("="*40)
    
    print("🖼️  Image Shapes 分布:")
    for shape, count in image_shapes.most_common():
        print(f"  - {str(shape):<15} : {count} 个切片")
        
    print("\n🏷️  Label Shapes 分布:")
    for shape, count in label_shapes.most_common():
        print(f"  - {str(shape):<15} : {count} 个切片")

if __name__ == "__main__":
    check_h5_shapes(H5_FOLDER_PATH)