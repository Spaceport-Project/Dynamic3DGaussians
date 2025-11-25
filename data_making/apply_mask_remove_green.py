import os
import numpy as np
import cv2

# Load the input image (assuming it will be provided)
input_image = cv2.imread('/home/hamit/DATA/processed_data/2025-03-27_15-46-48/colmap_input/input/0.png')  # or whatever the input filename will be

# Create the mask
mask = np.zeros((3000, 4096), dtype=np.uint8)
mask[300:2500, 1200:3100] = 255

# Convert mask to 3 channels to match the image
mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
folder = '/home/hamit/DATA/processed_data/2025-03-27_15-44-28/colmap_input/input_ai/'
output_folder = '/home/hamit/DATA/processed_data/2025-03-27_15-44-28/colmap_input/input_ai_masked_new_1'
os.makedirs(output_folder, exist_ok=True)
for file in os.listdir(folder):
    input_image = cv2.imread(os.path.join(folder, file))

    if input_image is None:
        print("Creating a sample image since input image was not found")
        # Create a sample colored image with a green background
        input_image = np.ones((3000, 4096, 3), dtype=np.uint8)
        # Set background to green
        input_image[:, :, 0] = 30  # blue component
        input_image[:, :, 1] = 180  # green component (higher for green background)
        input_image[:, :, 2] = 30  # red component
        
        # Add a non-green object in the center for demonstration
        center_y, center_x = 1500, 2048
        radius = 800
        y, x = np.ogrid[:3000, :4096]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        input_image[mask_circle, 0] = 100  # blue
        input_image[mask_circle, 1] = 50   # green (lower to make it non-green)
        input_image[mask_circle, 2] = 220  # red

    # Create the region mask
    region_mask = np.zeros((3000, 4096), dtype=np.uint8)
    # region_mask[0:2700, 1180:3400] = 255
    region_mask[500:2500, 1400:3000] = 255


    # Green screen removal (chroma keying)
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    
    # Define range for green color
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([80, 255, 255])
    
    # Create a mask for green pixels
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Invert the green mask (0 for green pixels, 255 for non-green)
    green_mask_inv = cv2.bitwise_not(green_mask)
    
    # Convert to 3 channel
    green_mask_inv_3ch = cv2.cvtColor(green_mask_inv, cv2.COLOR_GRAY2BGR)
    
    # Apply green removal mask (keep only non-green pixels)
    no_green_image = cv2.bitwise_and(input_image, green_mask_inv_3ch)
    
    # Apply region mask
    region_mask_3ch = cv2.cvtColor(region_mask, cv2.COLOR_GRAY2BGR)
    
    final_image = cv2.bitwise_and(no_green_image, region_mask_3ch)
   

# Save the masked result
    cv2.imwrite(os.path.join(output_folder, file), no_green_image)

    print("Masking applied to ",os.path.join(output_folder, file) )
print("Masked region (width):", "1180 to 3400")
print("Masked region (height):", "0 to 2700")