import cv2
import numpy as np

def find_blurry_regions(image_path, window_size=100, threshold=100, overlap=0.5):
    """
    Detect blurry regions in an image and return their coordinates.
    
    Parameters:
    - image_path: Path to the image file
    - window_size: Size of the sliding window (pixels)
    - threshold: Blur threshold (lower = more blurry)
    - overlap: Overlap ratio for sliding window (0-1)
    
    Returns:
    - List of dictionaries containing blurry region coordinates and scores
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not read image"}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Calculate step size based on overlap
    step = int(window_size * (1 - overlap))
    
    blurry_regions = []
    
    # Slide window across image
    for y in range(0, h - window_size + 1, step):
        for x in range(0, w - window_size + 1, step):
            # Extract window
            window = gray[y:y+window_size, x:x+window_size]
            
            # Calculate Laplacian variance for this window
            laplacian_var = cv2.Laplacian(window, cv2.CV_64F).var()
            
            # If blurry, store the region
            if laplacian_var < threshold:
                blurry_regions.append({
                    "x": x,
                    "y": y,
                    "width": window_size,
                    "height": window_size,
                    "blur_score": round(laplacian_var, 2)
                })
    
    return blurry_regions

def visualize_blurry_regions(image_path, blurry_regions, output_path="blur_map.jpg"):
    """
    Draw rectangles around blurry regions and save the result.
    """
    image = cv2.imread(image_path)
    
    for region in blurry_regions:
        x, y = region["x"], region["y"]
        w, h = region["width"], region["height"]
        score = region["blur_score"]
        
        # Draw rectangle (red for blurry)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Add blur score text
        cv2.putText(image, f"{score:.1f}", (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(output_path, image)
    print(f"Visualization saved to {output_path}")
    return image

# Example usage
image_path = "../im2_bck.png"

# Find blurry regions
blurry_regions = find_blurry_regions(image_path, window_size=500, threshold=100)

print(f"Found {len(blurry_regions)} blurry regions:")
for i, region in enumerate(blurry_regions, 1):
    print(f"Region {i}: x={region['x']}, y={region['y']}, "
          f"width={region['width']}, height={region['height']}, "
          f"blur_score={region['blur_score']}")

# Visualize the results
visualize_blurry_regions(image_path, blurry_regions)