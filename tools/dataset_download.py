import os, sys
# 🔴 必须在 import huggingface 之前设置镜像环境变量！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 🔴 强制清除可能存在的“死代理”，让镜像直连！
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# 现在再引入 requests 和 huggingface
import requests
from huggingface_hub import snapshot_download
import argparse

def test_mirror_connection():
    """测试当前服务器到底能不能连上镜像站"""
    print("🔍 正在测试与 hf-mirror.com 的网络连接...")
    try:
        res = requests.get("https://hf-mirror.com", timeout=5)
        if res.status_code == 200:
            print("✅ 镜像站连接成功！网络通畅。")
            return True
        else:
            print(f"⚠️ 镜像站连接异常，状态码: {res.status_code}")
            return False
    except Exception as e:
        print(f"❌ 镜像站连接彻底失败！原因: {str(e)}")
        print("💡 可能是你们学校/公司的内网彻底屏蔽了该网站，或者你需要特定的代理才能上网。")
        sys.exit(1)
        
def download_dataset(repo_name, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    print(f"\n🔄 开始下载数据集...")
    print(f"📦 目标仓库: {repo_name} ")
    print(f"💾 保存路径: {save_path}")
    
    try:
        # 使用 snapshot_download 下载整个数据集
        snapshot_download(
            repo_id=repo_name,
            repo_type="dataset",           # 必须指定为 dataset
            local_dir=save_path,
            local_dir_use_symlinks=False,  
            resume_download=True,          
            max_workers=8,                 
            ignore_patterns=[
                ".gitattributes", 
                "README.md", 
                "LICENSE", 
                ".DS_Store",       # 忽略根目录的 Mac 垃圾文件
                "*/.DS_Store",     # 忽略子文件夹的 Mac 垃圾文件
                "__MACOSX/*"       # 顺手把另一个常见的 Mac 垃圾文件夹也屏蔽掉
            ]
        )
        print(f"\n🎉 数据集已成功下载到: {save_path}")
        
    except Exception as e:
        print(f"\n❌ 最终下载失败，错误类型: {type(e).__name__}")
        print(f"详细报错: {str(e)}")
        # 打印引发这个报错的根本原因（非常重要）
        if e.__cause__:
            print(f"🔥 真正的底层网络错误是: {e.__cause__}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="limberc/LA2018", help="Dataset repository name")
    parser.add_argument("--save-path", type=str, required=True, help="Local path to save dataset")
    
    args = parser.parse_args()
    
    # 1. 先测试网络
    test_mirror_connection()
    # 2. 再开始下载
    download_dataset(args.repo, args.save_path)