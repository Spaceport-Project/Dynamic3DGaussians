import cv2
import numpy as np

# Load and threshold image
img = cv2.imread('mask_image8.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

# Distance transform
dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
max_dist = dist_transform.max()
print("Max distance in distance transform:", max_dist)

# Use a threshold that's a fraction of max_dist
# For example, keep only pixels that are at least 80% of max_dist
fraction = 0.8
threshold_value = max_dist * fraction
_, result = cv2.threshold(dist_transform, threshold_value, 255, cv2.THRESH_BINARY)
result = result.astype(np.uint8)

cv2.imwrite('band_removed_distance_final.png', result)
