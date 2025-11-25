import numpy as np
import cv2

# Create a black image of size 4096x3000
mask = np.zeros((3000, 4096), dtype=np.uint8)

# Define the masking region (white rectangle)
# Region: width from 1180 to 3400, height from 0 to 2700
mask[300:2600, 1100:3000] = 255

# Save the mask
cv2.imwrite('mask.png', mask)

print("Mask image has been created with dimensions:", mask.shape)
print("White region (width):", "1180 to 3400")
print("White region (height):", "0 to 2700")