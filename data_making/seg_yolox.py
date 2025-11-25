import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image  


# Load a model
model = YOLO("yolo11x-seg.yaml")

model = YOLO("yolo11x-seg.pt")

# Train the model
#results = model.train(data="coco8-seg.yaml",epochs=5)

# Evaluate model performance on the validation set  
#results = model.val()

# Perform object detection on an image
folder="/media/hamit/Elements/05-02-2025_Data/processed_data/2025-02-05_14-24-53_gain_8/ims_old/2/"
output_folder=os.path.join("masked_output")  
os.makedirs(output_folder, exist_ok=True)  

for k,file in enumerate(sorted(os.listdir(folder))):

    # if k < 34:
    #     continue
    file_name = os.path.join(folder, file)
    results = model(file_name, imgsz=(1280, 1280))

    # Get the original image
    orig_img = Image.open(file_name)
    img_array = np.array(orig_img)
    img_height, img_width = img_array.shape[:2]  
    alpha = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)  
    # For each detection in the image
    # For each detection in the image
    # results[0].show()

    
    for r in results[0]:
       
        # Check if the detection is a person (class 0)
        if hasattr(r, 'masks') and r.masks is not None and r.boxes is not None:
            # Get class IDs
            cls = r.boxes.cls.cpu().numpy()
            # Get boxes
            boxes = r.boxes.xyxy.cpu().numpy()
            # Get confidence scores
            conf = r.boxes.conf.cpu().numpy()
            # Get masks
            masks = r.masks.data.cpu().numpy()

            # Process only person detections (class 0)
            person_indices = np.where(cls == 0)[0]
            for idx in person_indices:
                # Print bounding box information
                box = boxes[idx]
                confidence = conf[idx]
                print(f"Person detected in {file}:")
                print(f"  Bounding Box: x1={box[0]:.2f}, y1={box[1]:.2f}, x2={box[2]:.2f}, y2={box[3]:.2f}")
                print(f"  Confidence: {confidence:.2f}")

                # Resize mask to match image dimensions
                mask = cv2.resize(masks[idx].astype(float), (img_width, img_height))
                # Convert mask to proper format
                mask = (mask * 255).astype(np.uint8)
                # Update alpha channel - make masked areas visible
                alpha = np.maximum(alpha, mask)
    # Create RGBA image
    rgba_img = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    if len(img_array.shape) == 2:  # Grayscale image
        rgba_img[..., :3] = np.stack([img_array] * 3, axis=-1)
    else:  # RGB image
        rgba_img[..., :3] = img_array[..., :3]  
    # Set alpha channel
    rgba_img[..., 3] = alpha
    output_img = Image.fromarray(rgba_img)
    output_path = os.path.join(output_folder, f"masked_{file}")
    output_img.save(output_path, format='PNG')
    # results[0].show()
    pass


# Export the model to ONNX format
#path = model.export(format="onnx")  # return path to exported model
