from pathlib import Path
import re
import h5py
import numpy as np
import glob
import os


def ims_index(path):
    # Get folder name just before the filename
    # .../ims/<number>/000001.png
    folder = os.path.basename(os.path.dirname(path))
    # Fallback: extract last integer in the path
    m = re.search(r'(\d+)', folder)
    return int(m.group(1)) if m else 0

# Get all npz files in the current directory
pattern="params*.npz"
dataset_path ='/home/hamit/Softwares/Dynamic3DGaussians/output/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850_0/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850'
output_path = '/home/hamit/Softwares/Dynamic3DGaussians/output/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850_0_combined/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850'
# npz_files = sorted(glob.glob('*.npz'))  # sorted to maintain order
npz_files = sorted(Path(dataset_path).glob(pattern), key= lambda x: int(x.stem.split("_")[-1]))
# npz_files = sorted(npz_files, key=ims_index)




# Process files in batches of 30
batch_size = 30
num_batches = (len(npz_files) + batch_size - 1) // batch_size  # ceiling division

# # for batch_idx in range(num_batches):
# #     start_idx = batch_idx * batch_size
# #     end_idx = min(start_idx + batch_size, len(npz_files))
# #     batch_files = npz_files[start_idx:end_idx]
    
# #     # Dictionary where each key maps to a list of arrays
# #     data_dict = {}
    
# #     for file in batch_files:
        
# #         try:
# #             data = np.load(file)
        
# #             for key in data.files:
# #                 if key != "seg_colors":
# #                     if key not in data_dict:
# #                         data_dict[key] = []
                    
# #                     if len(data[key].shape) == 3:
# #                         dat = np.squeeze(data[key])
# #                         data_dict[key].append(dat)
# #                     else:
# #                         data_dict[key].append(data[key])

# #         except Exception as e:
# #             print(f"Error {file} cannot open, {e}")
    
# #     # Convert lists to numpy object arrays
# #     for key in data_dict:
# #         data_dict[key] = np.array(data_dict[key], dtype=object)
    
# #     # Save this batch
# #     os.makedirs(output_path, exist_ok=True)

# #     np.savez(f'{output_path}/params_{batch_idx}.npz', **data_dict)
# #     print(f"Saved batch {batch_idx}: {len(batch_files)} files, keys: {list(data_dict.keys())}")


# for batch_idx in range((len(npz_files) + batch_size - 1) // batch_size):
#     start = batch_idx * batch_size
#     end = min(start + batch_size, len(npz_files))
#     batch_files = npz_files[start:end]
    
#     per_key_lists = {}
    
#     for file in batch_files:
#         try:
#             d = np.load(file)
#             for k in d.files:

#                 if k != "seg_colors":
#                     if len(d[k].shape) == 3:
#                         dat = np.squeeze(d[k])
#                         per_key_lists.setdefault(k, []).append(dat)  
#                     else:
#                         per_key_lists.setdefault(k, []).append(d[k])
#         except Exception as e:
#             print(f"Error {file} cannot open, {e}")
    
#     # Save to HDF5
#     with h5py.File(f'batch_{batch_idx}.h5', 'w') as f:
#         for k, arr_list in per_key_lists.items():
#             # Create a group for each key
#             grp = f.create_group(k)
#             for i, arr in enumerate(arr_list):
#                 grp.create_dataset(str(i), data=arr, compression='gzip')
    
#     print(f"Saved batch_{batch_idx}.h5 for files {start}..{end-1}")





# with h5py.File('/home/hamit/Softwares/Dynamic3DGaussians/output/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850_0_combined/2025-11-12_14-37-30_aliyenur_ahmet_1_50-1850/batch_0.h5', 'r') as f:
#     # Get array from file i for key 'positions'
#     # pos_i = f['positions']['5'][:]  # file index 5
    
#     # Or iterate through all files
#     for k in f.keys():
#         for i in range(len(f[k])):
#             arr = f[k][str(i)][:]
#             print(f"File {i}: shape {arr}")



for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(npz_files))
    batch_files = npz_files[start_idx:end_idx]
    
    # List to store dictionaries (one dict per file)
    files_list = []
    
    for file in batch_files:
        data = np.load(file)
        
        # Create a dictionary for this file
        file_dict = {}
        for key in data.files:
            try:
                if key != "seg_colors":
                    if len(data[key].shape) == 3:
                        dat = np.squeeze(data[key])
                        file_dict[key] = dat
                    else:
                        file_dict[key] = data[key]
            except Exception as e:
                print(f"Error {key} cannot open, {e}")
                break
        
        files_list.append(file_dict)
    
    # Convert to numpy object array
    files_array = np.array(files_list, dtype=object)
    
    # Save this batch
    np.savez(f'{output_path}/params_{batch_idx}.npz', files=files_array)
    print(f"Saved batch {batch_idx}: {len(batch_files)} files")

print(f"\nTotal: {len(npz_files)} files saved into {num_batches} batch files")