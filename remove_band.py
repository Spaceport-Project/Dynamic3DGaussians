import cv2
import numpy as np

# Load your image
img = cv2.imread('mask_image8.png', cv2.IMREAD_GRAYSCALE)

# Create binary image (adjust threshold as needed)
_, binary = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

# Method 1: Light erosion (recommended)
kernel = np.ones((4,4), np.uint8)
result1 = cv2.erode(binary, kernel, iterations=1)

# Method 2: Distance transform (alternative)
dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
_, result2 = cv2.threshold(dist_transform, 2, 255, cv2.THRESH_BINARY)
result2 = result2.astype(np.uint8)

# Save result
cv2.imwrite('band_removed1.png', result1)
cv2.imwrite('band_removed2.png', result2)

