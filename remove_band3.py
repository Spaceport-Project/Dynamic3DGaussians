import cv2
import numpy as np

# Load and threshold image
for i in range(2,43):

    img = cv2.imread(f'mask_image{i}.png', cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()

    # Use 30% of max distance to preserve internal regions
    threshold_value =  max_dist * 0.3  # This gives ~1.2 pixels
    _, result = cv2.threshold(dist_transform, threshold_value, 255, cv2.THRESH_BINARY)
    result = result.astype(np.uint8)

    cv2.imwrite(f'band_removed_optimal{i}.png', result)
