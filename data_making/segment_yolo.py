from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO('yolo11x-seg.pt',)  # or use 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt'

# Load your image
image_path = '/home/hamit/DATA/processed_data/2025-03-27_15-46-48/ims/0/000000.png'
image = cv2.imread(image_path)

# Run segmentation
results = model(image, imgsz=(1280, 1280))

# Process results
for result in results:
    # Get masks
    masks = result.masks.data.cpu().numpy()

    # Create empty mask image
    final_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Combine all masks
    for mask in masks:
        mask = mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        final_mask = cv2.bitwise_or(final_mask, mask)

    # Apply mask to original image
    masked_image = cv2.bitwise_and(image, image, mask=final_mask)

    # Save or display results
    cv2.imwrite('segmented_mask.jpg', final_mask)
    cv2.imwrite('masked_image.jpg', masked_image)