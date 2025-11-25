import cv2
import numpy as np
import matplotlib.pyplot as plt
# from google.colab import files

# Upload a sample image with purple tint
# print("Please upload your purple-tinted image")
# uploaded = 

# Get the filename of the uploaded image
filename = "/home/hamit/DATA/processed_data/2025-03-27_15-33-10/ims_black/0/000000.png"#list(uploaded.keys())[0]

# Read the image
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

# Display original image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img)

# Method to reduce purple tint
# Decrease red and blue channels (which make purple) and slightly boost green
def reduce_purple_tint(image, blue_factor=0.65, red_factor=0.65, green_boost=1.0):
    result = image.copy()
    # Adjust channels individually
    result[:,:,0] = np.clip(result[:,:,0] * red_factor, 0, 255).astype(np.uint8)  # Red channel
    result[:,:,1] = np.clip(result[:,:,1] * green_boost, 0, 255).astype(np.uint8)  # Green channel
    result[:,:,2] = np.clip(result[:,:,2] * blue_factor, 0, 255).astype(np.uint8)  # Blue channel
    return result
def adjust_colors(image, blue_factor=0.75, red_factor=0.7, green_factor=1.1):
    # Split the channels
    b, g, r = cv2.split(image)

    # Adjust each channel
    b = cv2.multiply(b, blue_factor)
    r = cv2.multiply(r, red_factor)
    g = cv2.multiply(g, green_factor)

    # Merge channels back
    adjusted = cv2.merge([b, g, r])
    return np.clip(adjusted, 0, 255).astype(np.uint8)

# Apply the correction
corrected_img = reduce_purple_tint(img)

# Display corrected image
plt.subplot(1, 2, 2)
plt.title('Corrected Image')
plt.imshow(corrected_img)
plt.tight_layout()
plt.show()

# Save the corrected image
output_filename = 'corrected_' + filename
cv2.imwrite(output_filename, cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR))
print(f"Corrected image saved as '{output_filename}'")

# You can adjust the factors in the reduce_purple_tint function to fine-tune the result