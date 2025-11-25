import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('/home/hamit/DATA/processed_data/2025-03-27_15-46-48/colmap_input/colmap_input_old/input_adjusted/13.png')

def adjust_colors(image, blue_factor=0.75, red_factor=0.7, green_factor=1.0):
    b, g, r = cv2.split(image)
    b = cv2.multiply(b, blue_factor)
    r = cv2.multiply(r, red_factor)
    g = cv2.multiply(g, green_factor)
    adjusted = cv2.merge([b, g, r])
    return np.clip(adjusted, 0, 255).astype(np.uint8)

# Method 1: Simple contrast adjustment using alpha beta
def adjust_contrast_brightness(image, alpha=1.3, beta=0.0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Method 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Apply color correction first

folder = '/home/hamit/DATA/processed_data/2025-03-27_15-46-48/colmap_input/input/'
output_folder = '/home/hamit/DATA/processed_data/2025-03-27_15-46-48/colmap_input/input_adjusted'
os.makedirs(output_folder, exist_ok=True)
for file in os.listdir(folder):
    input_image = cv2.imread(os.path.join(folder, file))
    color_corrected = adjust_colors(input_image)
    contrast_enhanced = adjust_contrast_brightness(color_corrected)
    cv2.imwrite(os.path.join(output_folder, file), contrast_enhanced)
# Apply different contrast methods
# contrast_enhanced = adjust_contrast_brightness(color_corrected)
# clahe_enhanced = apply_clahe(color_corrected)

#