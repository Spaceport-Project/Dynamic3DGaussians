import multiprocessing
from PIL import Image
import os

def resize_images_in_folder(folder_path, output_folder, new_width, new_height):
  # Ensure the output folder exists
    # if not os.path.exists(output_folder):
    #   os.makedirs(output_folder)

  # Iterate over all files in the folder
#   for folder in os.listdir(folder_path):
#       if os.path.isdir(os.path.join(folder_path, folder)):
    for filename in os.listdir(folder_path):
        folder=os.path.basename(folder_path)
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                # Calculate the new size maintaining the aspect ratio
                img = img.resize((new_width, new_height))
                
                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, folder, filename)
                
                if not os.path.exists(os.path.join(output_folder, folder)):
                    os.makedirs(os.path.join(output_folder,folder))
                
                img.save(output_path, "PNG")
                print(f"Resized and saved {filename} to {output_folder}/{folder}")


    

# Example usage
folder_path = '/media/hamit/HamitsKingston/calibs'
output_folder = '/media/hamit/HamitsKingston/calibs_resized'
new_width = 2048  # Desired width
new_height = 1500  # Desired height

folder_list=[]
# for folder in os.listdir(folder_path):
#     if os.path.isdir(os.path.join(folder_path, folder)):
#         folder_list.append((os.path.join(folder_path, folder), output_folder, new_width, new_height))

folder_list.append((os.path.join(folder_path, "Cam_005"),output_folder, new_width, new_height))
with multiprocessing.Pool(processes=12) as pool_resize:
    pool_resize.starmap(resize_images_in_folder, folder_list)


# resize_images_in_folder(folder_path, output_folder, new_width, new_height)