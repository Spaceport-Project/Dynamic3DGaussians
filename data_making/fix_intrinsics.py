import yaml
import os
import pprint

def create_intrinsics(folder_path):
    params_dict = {}
    for yml_file in os.listdir(folder_path):
        root, _ = os.path.splitext(yml_file)
        with open(os.path.join(folder_path, yml_file), 'r') as file:
            data = yaml.safe_load(file)
            # print(data['camera_matrix']['data'])
            params = [ pr for pr in data['camera_matrix']['data'] if pr !=0 and pr != 1 ]
            params[1], params[2] = params[2], params[1]
            params = params + data['distortion_coefficients']['data']
            params_dict[int(root)] = params
            
    # print(params_dict) 
    pprint.pprint(params_dict)
    return params_dict
   
if __name__ == "__main__":
    create_intrinsics("/home/hamit/basler_camera_calibrations/calibrations_list/")