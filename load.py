import os 
import shutil
from pathlib import Path

def import_file(src_path, target_dir):
    """
    Imports a file from src_path to target_dir.
    If target_dir does not exist, it is created.
    If a file with the same name exists in target_dir, it is overwritten.
    
    Parameters:
    src_path (str or Path): The source file path.
    target_dir (str or Path): The destination directory path.
    
    Returns:
    """
    src_path = Path(src_path)
    target_dir = Path(target_dir)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file '{src_path}' does not exist.")
    
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        
    target_path = target_dir / src_path.name
    shutil.copy2(src_path, target_path)
    return target_path

if __name__ == "__main__":
    # 导入单个文件示例
    result = import_file("C:\\Users\\caoran\\Desktop\\大明天下.txt", "raw_novels/")
    print(f"文件已导入到: {result}")