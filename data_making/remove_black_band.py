import cv2
import numpy as np

def remove_transition_band(image_path, output_path=None):
    """
    Remove the dark gray transition band around a person and replace with pure black
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert to RGB for easier processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a mask for the transition band
    # We want to identify pixels that are dark but not pure black
    # and not part of the main subject
    
    # Define the range for the transition band (around RGB 30,30,30 to black)
    lower_bound = np.array([0, 0, 0])      # Pure black
    upper_bound = np.array([15, 15, 15])   # Slightly above RGB 30,30,30
    
    # Create mask for dark pixels in the transition range
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3,3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Create the result image
    result = img_rgb.copy()
    
    # Replace the transition band pixels with pure black
    result[mask > 0] = [0, 0, 0]
    
    # Convert back to BGR for saving
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    if output_path:
        cv2.imwrite(output_path, result_bgr)
        print(f"Processed image saved to: {output_path}")
    
    return result_bgr

# Alternative approach with more precise control
def remove_transition_band_advanced(image_path, output_path=None, threshold_low=30, threshold_high=15):
    """
    Advanced version with adjustable thresholds
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert to grayscale to identify dark regions
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create mask for pixels in the transition range
    mask = (gray > 0) & (gray <= threshold_high)
    
    # Apply additional filtering to avoid affecting the main subject
    # Use edge detection to preserve subject boundaries
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    
    # Exclude edge areas from the mask to preserve subject details
    mask = mask & (edges_dilated == 0)
    
    # Create result
    result = img.copy()
    result[mask] = [0, 0, 0]  # Set to pure black
    
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Advanced processed image saved to: {output_path}")
    
    return result

# Example usage (you'll need to replace with your actual image path)

remove_transition_band('/home/hamit/Softwares/Dynamic3DGaussians/im_masked.png', 'output.png')  
remove_transition_band_advanced('/home/hamit/Softwares/Dynamic3DGaussians/im_masked.png', 'output2.png', threshold_low=30, threshold_high=50)  
  
# print("OpenCV transition band removal functions created!")
# print("\nUsage examples:")
# print("1. Basic: remove_transition_band('your_image.jpg', 'output.jpg')")
# print("2. Advanced: remove_transition_band_advanced('your_image.jpg', 'output.jpg', threshold_low=30, threshold_high=50)")
# print("\nThe functions will:")
# print("- Identify pixels in the dark gray transition range")
# print("- Replace them with pure black (0,0,0)")
# print("- Preserve the main subject details")