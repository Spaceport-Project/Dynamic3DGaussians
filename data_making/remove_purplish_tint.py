import cv2
import numpy as np
import os
import argparse



def adjust_colors(image, blue_factor=0.75, red_factor=0.7, green_factor=1.1):
    # Split the channels
    b, g, r = cv2.split(image)

    # Adjust each channel
    b = cv2.multiply(b, blue_factor)
    r = cv2.multiply(r, red_factor)
    g = cv2.multiply(g, green_factor)

    # Merge channels back
    adjusted = cv2.merge([b, g, r])
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def reduce_purple_tint(image, blue_factor=0.8, red_factor=0.6, green_boost=1.):
    result = image.copy()
    # Adjust channels individually
    result[:,:,0] = np.clip(result[:,:,0] * red_factor, 0, 255).astype(np.uint8)  # Red channel
    result[:,:,1] = np.clip(result[:,:,1] * green_boost, 0, 255).astype(np.uint8)  # Green channel
    result[:,:,2] = np.clip(result[:,:,2] * blue_factor, 0, 255).astype(np.uint8)  # Blue channel
    return result


def reduce_purple_tint_for_folder(folder_name):
    for file in sorted(os.listdir(folder_name)):
        if file.endswith("png"):
            img = cv2.imread(os.path.join(folder_name, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output_path = os.path.join(folder_name, file)
            output_path = output_path.replace("ims", "ims_new")
            # print(os.path.dirname(output_path))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            res = adjust_colors(img)
           # res = reduce_purple_tint(img)
            cv2.imwrite(output_path, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="remove purplish tint from images")
    parser.add_argument("--path", required=True, type=str, help="Path to the folder")
    args = parser.parse_args()

    reduce_purple_tint_for_folder(args.path)