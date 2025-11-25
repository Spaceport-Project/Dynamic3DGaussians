import numpy as np
import os
import torch
REMOVE_BACKGROUND = False  # False or True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data_save_as_files(folder_path, output_folder):
        params_files =[ os.path.join(folder_path, file) for file in  os.listdir(folder_path) if file.startswith("params")]
        params_files = sorted(params_files, key= lambda x : int(os.path.basename(x).split("_")[1].split(".")[0]) if len(os.path.basename(x).split("_")) > 1 else os.path.basename(x).split("_")[0].split(".")[0]) 

        
        scene_data = []
        total_length = 0
        cnt = 1
        per_frames = 30
        iter = 44
        tot_cnt = 0
        os.makedirs(output_folder, exist_ok=True)


        for l, params_file  in enumerate(params_files):
        
            params = dict(np.load(params_file))  
            print(f"{params_file} loaded!")

            # params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
            # is_fg = params['seg_colors'][:, 0] > 0.5
        
            length = len(params['means3D'])
            total_length = total_length + length
            ind = 0
            print(f"length, total timesteps:", length, total_length, l)
            s=0
            scene_data.append([])
            for t in range(length): #len(params['means3D'])):

                # if 587 < tot_cnt < 600 or 288 <tot_cnt < 295 or 374 < tot_cnt < 390:
                #     tot_cnt += 1
                #     continue
                # if 562 < tot_cnt < 582:
                #     tot_cnt += 1
                #     continue

                # rendervar = { k: params[k][t] for k in params.keys()}
                if ind == 0 :
                    rendervar = {
                        'means3D': params['means3D'][s],
                        'rgb_colors': (params['rgb_colors'][s]),
                        'unnorm_rotations': params['unnorm_rotations'][s],
                        'logit_opacities': params['logit_opacities'][s],
                        'log_scales': params['log_scales'][s]
                    
                    }
                else:
                    #  rendervar = {
                    #     'means3D': params['means3D'][s],
                    #     'rgb_colors': params['rgb_colors'][s],
                    #     'unnorm_rotations': params['unnorm_rotations'][s],
                       
                    
                    # }
                      rendervar = {
                        'means3D': params['means3D'][s],
                        'rgb_colors': params['rgb_colors'][s],
                        'unnorm_rotations': params['unnorm_rotations'][s],
                        'logit_opacities': params['logit_opacities'][s],
                        'log_scales': params['log_scales'][s]
                    
                    }

                if s == 0:
                    s = t
                
                scene_data[-1].append(rendervar)
                if cnt % per_frames  == 0 :
                    array = np.array([subscene for subscene in [scene for scene in scene_data]], dtype=object)
                    np.savez(f"{output_folder}/params_{iter}.npz", array, allow_pickle=True)
                    iter += 1
                    scene_data = []
                    scene_data.append([])
                    ind = -1
                    # if length > per_frames:
                    #     ind = 0
                    #     s = 0
                    #     cnt += 1
                    #     continue
                tot_cnt +=1        
                ind += 1
                s += 1
                cnt += 1
        
        if (cnt -1) % per_frames != 0:
            array = np.array([subscene for subscene in [scene for scene in scene_data]], dtype=object)
            np.savez(f"{output_folder}/params_{iter}.npz", array, allow_pickle=True)


          
        # return scene_data
def print_data(output_folder):
    params_files =[ os.path.join(output_folder, file) for file in  os.listdir(output_folder) if file.startswith("params")]

    params_files = sorted(params_files, key= lambda x : int(os.path.basename(x).split("_")[1].split(".")[0]) if len(os.path.basename(x).split("_")) > 1 else os.path.basename(x).split("_")[0].split(".")[0]) 
    for l, params_file  in enumerate(params_files):
        
        data = np.load(params_file, allow_pickle=True)
        print(params_file)
        for dat in data:
            if dat == "allow_pickle":
                continue
            print(len(data[dat]))
            da = data[dat]
            pass
            # print(data[dat])

def save_params(output_params, output_folder):
  
    # to_save = {}
    # iter = 0
    # for k in output_params[0].keys():
    #     if k in output_params[1].keys():
    #         # to_save[k] = np.stack([params[k] for params in output_params])
    #         arr = []
    #         for l, params in enumerate(output_params):
    #              arr.append(params[k])
                 
    #         to_save[k] = np.array(arr, dtype=object) 
    #         # to_save[k]= np.array([params[k] for l, params in enumerate(output_params)], dtype=object)
    #     else:
    #         to_save[k] = output_params[0][k]
    os.makedirs(output_folder, exist_ok=True)
    # np.savez(f"{output_folder}/params_{iter}.npz", **to_save)
    l = 0
    iter = 0
    
       
    while l < len(output_params) :
        to_save = {}
        s = l + 30 if len(output_params) > l + 30 else len(output_params)
       
        
        for k in output_params[0].keys():
            if k in output_params[1].keys():
            
          
            
                to_save[k]= np.array([params[k] for params in output_params[l:s]], dtype=object)
            elif l == 0:
                to_save[k]= output_params[0][k]
           
        np.savez(f"{output_folder}/params_{iter}.npz", **to_save, allow_pickle=True)
        l += 30
        iter +=1 


if __name__ == "__main__":
    source_path = "/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-38-56_3412x2500_combin3_2_48sc_test1_metashape_contrast11_sharp05_529/2025-08-06_16-38-56_3412x2500_combin3_2_48sc/"

    target_path ="/home/hamit/Softwares/Dynamic3DGaussians/output/2025-08-06_16-38-56_3412x2500_combin3_2_48sc_test1_metashape_contrast11_sharp05_50_multiprocess/2025-08-06_16-38-56_3412x2500_combin3_2_48sc/"
    
    load_data_save_as_files(source_path, target_path)
    
    print_data(target_path)
    # data = np.load('output/samples/sample_1/params_0.npz', allow_pickle=True)
    # for dat in data:
    #     print(data[dat])
    # dat = data["arr_0"]
    pass