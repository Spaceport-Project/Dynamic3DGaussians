from argparse import ArgumentParser
import os
import cv2

def resize_image(input_path, output_path, dim):
  # Read the image from the file
  image = cv2.imread(input_path)
  
  # Check if the image was successfully loaded
  if image is None:
      print(f"Error: Unable to load image from {input_path}")
      return
  
  # Resize the image
  resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
  # Save the resized image to the output path
  cv2.imwrite(output_path, resized_image)
  print(f"Resized image saved to {output_path}")

# Example usage
if __name__ == '__main__':

    
    # parser = ArgumentParser("Colmap converter")
    # parser.add_argument("--input_image", required=True, type=str)
    # parser.add_argument("--output_image", required=True, type=str)
    # parser.add_argument("--dim", required=True, type=str)
    # args = parser.parse_args()
    # width, height = args.dim.split('x')

    source = "/home/hamit/DATA/19-12-2024_Data/2024-12-19_19-12-14_4096/processed_data_300-650/seg"
    target = "/home/hamit/DATA/19-12-2024_Data/2024-12-19_19-12-14_4096/processed_data_300-650/seg_resized"

    for folder in os.listdir(source):
      if os.path.isdir(os.path.join(source, folder)):
        os.makedirs(os.path.join(target, folder))
        for file in os.listdir(os.path.join(source, folder)):
          resize_image(os.path.join(source, folder, file),  os.path.join(target, folder, file), (2108,1524))
    
    # resize_image(args.input_image, args.output_image, (int(width), int(height)))