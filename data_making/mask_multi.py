import cv2
import numpy as np
import multiprocessing
import os
import argparse
from PIL import Image, ImageFile



def prepare_data(path, input_folder, output_folder, output_seg_folder=None):
    tuple_list = []
    for file in os.listdir(path):
        img_index = os.path.basename(file).split(".")[0]
        for fol in os.listdir(os.path.join(input_folder)):
            
            if fol == img_index and  os.path.isdir(os.path.join(input_folder, fol)):
                input_path = os.path.join(input_folder, img_index) 
                output_path = os.path.join(output_folder, img_index ) 
                if output_seg_folder is not None:
                    output_seg_path = os.path.join(output_seg_folder, img_index)  
                else:
                   output_seg_path=None      
                tuple_list.append((os.path.join(path, file), input_path, output_path, output_seg_path))        
      

    return tuple_list

def mask_images(file_path, input_folder, output_folder, output_seg_folder):
    image = cv2.imread(file_path, -1)
    alpha_exist = True
    # Split the channels
    if image.shape[2] == 4: 
        b, g, r, alpha = cv2.split(image)
        mask = np.zeros_like(alpha)
        mask[alpha == 0] = 255
    else:
        alpha_exist=False
    
   
    os.makedirs(output_folder, exist_ok=True)
    if output_seg_folder is not None:
        os.makedirs(output_seg_folder, exist_ok=True)


    for img in os.listdir(input_folder):   
        target_img = cv2.imread(os.path.join(input_folder, img))
        #target_img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  
        width, height = target_img.shape[1], target_img.shape[0]
        seg = Image.new("RGB", (width, height), "white")  
        seg = np.array(seg)
        if alpha_exist:
        # Make sure mask and target image have same dimensions
            mask = cv2.resize(mask, (target_img.shape[1], target_img.shape[0]))

            # Apply mask to target image
            masked_img = cv2.bitwise_and(target_img, target_img, mask=cv2.bitwise_not(mask))
            masked_seg = cv2.bitwise_and(seg, seg, mask=cv2.bitwise_not(mask))
            cv2.imwrite(os.path.join(output_folder, img), masked_img)
            if output_seg_folder is not None :
                cv2.imwrite(os.path.join(output_seg_folder, img), masked_seg)
        else:
            cv2.imwrite(os.path.join(output_folder, img), target_img)
            if output_seg_folder is not None :
                cv2.imwrite(os.path.join(output_seg_folder, img), seg)
    print(f"Done! {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="masking images")
    parser.add_argument("--path", required=True, type=str, help="Path to the folder to masks")
    parser.add_argument("--input_folder", required=True, type=str, help="Path to the input  folder")
    parser.add_argument("--output_folder", required=True, type=str, help="Path to the output folder")
    parser.add_argument("--output_seg_folder", default=None, help="Path to the output folder")

    args = parser.parse_args()


    tuple_list=prepare_data(args.path, args.input_folder, args.output_folder, args.output_seg_folder)
    print(tuple_list)
    with multiprocessing.Pool(processes=4) as pool_masking:
            
        results = []  
        
        for lst in tuple_list:
            result = pool_masking.apply_async(mask_images, lst)  
            results.append(result)  
        results = [r.get() for r in results] 
                
        print("Masking Results:", results)  
