import os
import numpy as np
import cv2

# Load the input image (assuming it will be provided)
input_image = cv2.imread('/home/hamit/DATA/processed_data/2025-03-27_15-46-48/colmap_input/input/0.png')  # or whatever the input filename will be

# Create the mask
mask = np.zeros((3000, 4096), dtype=np.uint8)
mask[500:2500, 1400:3000] = 255

# Convert mask to 3 channels to match the image
mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
folder = '/home/hamit/DATA/processed_data/2025-03-27_15-46-48/colmap_input/input/'
output_folder = '/home/hamit/DATA/processed_data/2025-03-27_15-46-48/colmap_input/input_masked'
os.makedirs(output_folder, exist_ok=True)
for file in os.listdir(folder):
    input_image = cv2.imread(os.path.join(folder, file))

# Apply the mask (everything outside the white region becomes black)
    masked_image = cv2.bitwise_and(input_image, mask_3channel)

# Save the masked result
    cv2.imwrite(os.path.join(output_folder, file), masked_image)

    print("Masking applied to ",os.path.join(output_folder, file) )
print("Masked region (width):", "1180 to 3400")
print("Masked region (height):", "0 to 2700")