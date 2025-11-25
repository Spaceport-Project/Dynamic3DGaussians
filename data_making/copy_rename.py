import os
import shutil
from pathlib import Path
def get_param_index(filename):  
    """  
    Extract numeric index from 'params_{n}.npz'  
    """  
    base = filename[len('params_'):-len('.npz')]  # strip prefix and suffix  
    return int(base)  
def copy_and_rename_params(folder1, folder2):
    """
    Copy params files from folder2 to folder1 with sequential numbering.
    
    Args:
        folder1: Path to the first folder (destination)
        folder2: Path to the second folder (source)
    """
    # Get all params_*.npz files from folder1
    folder1_files = [f for f in os.listdir(folder1) if f.startswith('params_') and f.endswith('.npz')]
    # folder1_files.sort(key=get_param_index)  

    # Find the highest number in folder1 using numeric sort
    if folder1_files:
        max_num = max(get_param_index(f) for f in folder1_files)
    else:
        max_num = -1
    
    # Get all params_*.npz files from folder2
# Get all params_*.npz files from folder2 and sort by numeric index
    folder2_files = [f for f in os.listdir(folder2)
                     if f.startswith('params_') and f.endswith('.npz')]
    folder2_files.sort(key=get_param_index)

    # Copy files from folder2 to folder1 with new numbering
    for i, file in enumerate(folder2_files):
        new_num = max_num + 1 + i
        new_name = f'params_{new_num}.npz'
        
        src = os.path.join(folder2, file)
        dst = os.path.join(folder1, new_name)
        
        shutil.copy2(src, dst)
        print(f'Copied {file} -> {new_name}')
    
    print(f'\nTotal files copied: {len(folder2_files)}')

# Usage
folder1_path = '/home/hamit/Softwares/Dynamic3DGaussians/output/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850_0/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850'
folder2_path = '/home/hamit/Softwares/Dynamic3DGaussians/output/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850_542/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850'

copy_and_rename_params(folder1_path, folder2_path)