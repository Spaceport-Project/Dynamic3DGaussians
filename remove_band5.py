import cv2
import numpy as np

def remove_white_band(image_path, threshold=30):
    # Load and threshold the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result = np.zeros_like(binary_img)
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Keep large external contours (main body)
        if hierarchy[0][i][3] == -1 and area > 2000:
            cv2.fillPoly(result, [contour], 255)
        
        # Keep internal features
        elif hierarchy[0][i][3] != -1 and area > 100:
            cv2.fillPoly(result, [contour], 255)
        
        # Keep compact medium regions
        elif area > 300:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0 and area / (perimeter * perimeter) > 0.01:
                cv2.fillPoly(result, [contour], 255)
    
    return result

# Usage:




def remove_all_white_bands(image_path, threshold=30, distance_threshold=4):
    """Ultra aggressive white band removal"""
    
    # Load and threshold
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Distance transform method - most effective
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    _, result = cv2.threshold(dist_transform, distance_threshold, 255, cv2.THRESH_BINARY)
    
    return result.astype(np.uint8)

# Alternative: Heavy erosion method
def remove_bands_erosion(image_path, threshold=30, erosion_iterations=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.erode(binary_img, kernel, iterations=erosion_iterations)
    
    return result

def nuclear_band_elimination(image_path, threshold=30):
    """Nuclear option - eliminate EVERY thin band"""
    
    # Load and threshold
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # NUCLEAR OPTION 1: Distance transform 4.5 (RECOMMENDED)
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    _, result = cv2.threshold(dist_transform, 4.5, 255, cv2.THRESH_BINARY)
    
    # Clean up any tiny remaining specs
    contours, _ = cv2.findContours(result.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_result = np.zeros_like(binary_img)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2:  # Remove tiny specs
            cv2.fillPoly(final_result, [contour], 255)
    
    return final_result

# Alternative nuclear option (if you want slightly more content):
def nuclear_erosion_method(image_path, threshold=30):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.erode(binary_img, kernel, iterations=4)

    dist_transform = cv2.distanceTransform(result, cv2.DIST_L2, 5)
    min_diameter_pixels=2
    # Keep only regions where radius >= min_diameter/2
    min_radius = min_diameter_pixels / 2.0
    _, thick_regions = cv2.threshold(dist_transform, min_radius, 255, cv2.THRESH_BINARY)
    
    return result, thick_regions.astype(np.uint8)

def keep_thick_regions(image_path, min_diameter_pixels=8, threshold=180):
    """
    Keep only white regions with diameter >= min_diameter_pixels
    
    Args:
        image_path: Path to your image
        min_diameter_pixels: Minimum diameter to keep (recommended: 4-8)
        threshold: Grayscale threshold for binarization
    """
    
    # Load and threshold image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Distance transform gives radius at each point
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    
    # Keep only regions where radius >= min_diameter/2
    min_radius = min_diameter_pixels / 2.0
    _, thick_regions = cv2.threshold(dist_transform, min_radius, 255, cv2.THRESH_BINARY)
    
    return thick_regions.astype(np.uint8)

def keep_top2_by_radius(image_path, min_diameter_pixels=6, threshold=120, output_path='top2_regions.png'):
    # Load and threshold image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Distance transform for thickness
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    min_radius = min_diameter_pixels / 2.0
    _, thick_regions = cv2.threshold(dist_transform, min_radius, 255, cv2.THRESH_BINARY)
    thick_regions = thick_regions.astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thick_regions, connectivity=8)
    
    # Gather region info
    regions = []
    for label in range(1, num_labels):  # label 0 is background
        mask = (labels == label).astype(np.uint8) * 255
        region_dist = dist_transform[labels == label]
        max_radius = np.max(region_dist) if region_dist.size > 0 else 0
        regions.append({
            'label': label,
            'max_radius': max_radius,
            'mask': mask
        })
    
    # Sort by max_radius (descending)
    regions_sorted = sorted(regions, key=lambda r: r['max_radius'], reverse=True)
    
    # Combine top 2 masks
    combined = np.zeros_like(binary_img)
    for region in regions_sorted[:3]:
        combined = cv2.bitwise_or(combined, region['mask'])
    
    # Save the result
    cv2.imwrite(output_path, combined)
    print(f"Saved top 2 thickest regions to {output_path}")

def remove_white_band_morphological(image_path, band_thickness=10):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_thickness*2+1, band_thickness*2+1))
    eroded = cv2.erode(image, kernel, iterations=1)
    restored = cv2.dilate(eroded, kernel, iterations=1)
    return restored

def remove_white_band(image_path, band_thickness=1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()

    for contour in contours:
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        kernel = np.ones((band_thickness, band_thickness), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        band_mask = cv2.subtract(mask, eroded_mask)

        result[band_mask == 255] = 0

    return result

def remove_white_band2(image_path):

    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold to ensure binary image
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the outer band)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask = np.ones_like(img) * 255  # Start with all white
    cv2.drawContours(mask, [largest_contour], -1, 0, thickness=cv2.FILLED)  # Fill the largest contour with black

    # Apply the mask to the original image
    result = cv2.bitwise_and(img, mask)
    return result

def remove_white_band3(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold to ensure binary image
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(img)

    # Loop through all contours
    for i, h in enumerate(hierarchy[0]):
        # If the contour has a parent (h[3] != -1), it's a hole inside the outer band
        # if h[3] != -1:
        #     cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
        if h[3] == 0:  # Parent is the outermost contour  
            cv2.drawContours(mask, contours, i, 255, cv2.FILLED) 
    return mask
def remove_white_band4(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Choose a kernel size roughly equal to the thickness of the band
    kernel_size = 25  # Adjust this value as needed!
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Erode the image to remove the outer band
    eroded = cv2.erode(binary, kernel, iterations=1)

    # Find the largest contour (outer band)
   
    return eroded

    

    # Save or display the result
# cv2.imshow('Result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for i in range(0,43):
#     ultra_clean = keep_top2_by_radius(f'mask_image{i}.png', min_diameter_pixels=4, output_path=f'no_bands_result{i}.png')
result = remove_white_band4("mask_image_9.png")  
cv2.imwrite('no_bands_result.png', result)
    # cv2.imwrite(f'no_bands_result{i}.png', ultra_clean)
    # cv2.imwrite(f'no_bands_result_2_{i}.png', ultra_clean2)

    # result = remove_white_band('mask_image8.png')
    # cv2.imwrite('cleaned_result.png', result)