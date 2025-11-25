import cv2
import numpy as np

# Load and threshold image
for i in range(2,43):
    img = cv2.imread(f'mask_image{i}.png', cv2.IMREAD_GRAYSCALE)
    target_removal_ratio =0.9    
    result = img.copy()
    original_pixels = np.sum(img == 255)
    target_pixels = int(original_pixels * (1 - target_removal_ratio))

    kernel = np.ones((3,3), np.uint8)

    while np.sum(result == 255) > target_pixels and np.sum(result == 255) > 50:
        result = cv2.erode(result, kernel, iterations=1)
        if np.sum(result == 255) == 0:  # Prevent complete removal
            break

    cv2.imwrite(f"result_image_{i}.png",result)
