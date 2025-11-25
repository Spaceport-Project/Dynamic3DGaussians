import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import warnings  

# Suppress the specific warning about antialias  
warnings.filterwarnings('ignore', message='The default value of the antialias parameter.*')  

def mask_person(image_tensor, green_rgb, threshold=0.1):
    green_tensor = torch.tensor(green_rgb, device=image_tensor.device).view(1, 3, 1, 1)
    diff = torch.abs(image_tensor - green_tensor)
    mask = torch.sum(diff, dim=1) > threshold
    return mask

def crop_person(image_tensor, mask):
    # Get indices where mask is True (non-zero)
    non_zero_coords = torch.nonzero(mask.squeeze())  # Remove batch dimension

    # Extract min and max coordinates correctly
    y_coords = non_zero_coords[:, 0]  # First column contains y coordinates
    x_coords = non_zero_coords[:, 1]  # Second column contains x coordinates

    y_min = torch.min(y_coords)
    y_max = torch.max(y_coords)
    x_min = torch.min(x_coords)
    x_max = torch.max(x_coords)

    # Crop the image using the coordinates
    cropped_image = image_tensor[:, :, y_min:y_max+1, x_min:x_max+1]
    return cropped_image, (y_min, y_max, x_min, x_max)

def resize_and_place2(image_tensor, cropped_tensor, bbox, scale=2):
    y_min, y_max, x_min, x_max = bbox

    # Resize the cropped person
    resize_transform = T.Resize(
        (int(cropped_tensor.shape[2] * scale),
         int(cropped_tensor.shape[3] * scale)),
        antialias=True
    )
    resized_person = resize_transform(cropped_tensor)

    new_height, new_width = resized_person.shape[2], resized_person.shape[3]

    # Calculate x position (centered horizontally)
    center_x = (x_min + x_max) // 2
    x_start = max(0, center_x - new_width // 2)

    # Calculate y position (bottom-aligned with original crop)
    y_start = y_max - new_height  # This will place the bottom of resized person at y_max

    # Adjust if the resized person would go outside the image bounds
    if y_start < 0:
        y_start = 0
    if x_start + new_width > image_tensor.shape[3]:
        x_start = image_tensor.shape[3] - new_width
    x_start = max(0, x_start)

    y_end = min(y_start + new_height, image_tensor.shape[2])
    x_end = min(x_start + new_width, image_tensor.shape[3])

    # Create output image
    output_image = image_tensor.clone()

    # Calculate the actual heights and widths to copy
    height_to_copy = min(new_height, y_end - y_start)
    width_to_copy = min(new_width, x_end - x_start)

    # Only copy the portion that fits within the image bounds
    output_image[:, :, y_start:y_start+height_to_copy, x_start:x_start+width_to_copy] = \
        resized_person[:, :, :height_to_copy, :width_to_copy]

    return output_image

def resize_and_place(image_tensor, cropped_tensor, bbox, scale=2):
    y_min, y_max, x_min, x_max = bbox

    # Resize the cropped person
    resize_transform = T.Resize((int(cropped_tensor.shape[2] * scale),
                               int(cropped_tensor.shape[3] * scale)))
    resized_person = resize_transform(cropped_tensor)

    # Calculate new placement coordinates
    new_height, new_width = resized_person.shape[2], resized_person.shape[3]
    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    # Calculate starting positions ensuring they're within image bounds
    y_start = max(0, center_y - new_height // 2)
    x_start = max(0, center_x - new_width // 2)

    # Adjust if the resized person would go outside the image bounds
    y_start = min(y_start, image_tensor.shape[2] - new_height)
    x_start = min(x_start, image_tensor.shape[3] - new_width)

    # Ensure positive coordinates
    y_start = max(0, y_start)
    x_start = max(0, x_start)

    y_end = min(y_start + new_height, image_tensor.shape[2])
    x_end = min(x_start + new_width, image_tensor.shape[3])

    # Create output image
    output_image = image_tensor.clone()

    # Only copy the portion that fits within the image bounds
    height_to_copy = min(new_height, y_end - y_start)
    width_to_copy = min(new_width, x_end - x_start)

    output_image[:, :, y_start:y_start+height_to_copy, x_start:x_start+width_to_copy] = \
        resized_person[:, :, :height_to_copy, :width_to_copy]

    return output_image

def resize_and_place3(image_tensor, cropped_tensor, bbox, scale=2):
    y_min, y_max, x_min, x_max = bbox

    # Resize the cropped person
    resize_transform = T.Resize(
        (int(cropped_tensor.shape[2] * scale),
         int(cropped_tensor.shape[3] * scale)),
        antialias=True
    )
    resized_person = resize_transform(cropped_tensor)

    new_height, new_width = resized_person.shape[2], resized_person.shape[3]

    # Set x position to x_min (left-aligned)
    x_start = x_min

    # Set y position (bottom-aligned with original crop)
    y_start = y_max - new_height  # This will place the bottom of resized person at y_max

    # Adjust if the resized person would go outside the image bounds
    if y_start < 0:
        y_start = 0
    if x_start + new_width > image_tensor.shape[3]:
        new_width = image_tensor.shape[3] - x_start

    y_end = min(y_start + new_height, image_tensor.shape[2])
    x_end = min(x_start + new_width, image_tensor.shape[3])

    # Create output image
    output_image = image_tensor.clone()

    # Calculate the actual heights and widths to copy
    height_to_copy = min(new_height, y_end - y_start)
    width_to_copy = min(new_width, x_end - x_start)

    # Only copy the portion that fits within the image bounds
    output_image[:, :, y_start:y_start+height_to_copy, x_start:x_start+width_to_copy] = \
        resized_person[:, :, :height_to_copy, :width_to_copy]

    return output_image
def process_image(image_path, green_rgb, scale=2, threshold=0.1):
    # Load the image and move it to GPU
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    # Mask the person
    mask = mask_person(image_tensor, green_rgb, threshold)

    # Crop the person
    cropped_tensor, bbox = crop_person(image_tensor, mask)

    # Resize and place the person back
    output_tensor = resize_and_place2(image_tensor, cropped_tensor, bbox, scale)

    # Convert back to PIL image for saving
    output_image = T.ToPILImage()(output_tensor.squeeze(0).cpu())
    return output_image

# Example usage
if __name__ == "__main__":
    green_rgb = [0, 177.0 / 255, 64.0 / 255]
    image_path = "00001.png"

    try:
        output_image = process_image(image_path, green_rgb, scale=1.5)
        output_image.save("output_image2.png")
        print("Output image saved as output_image2.png")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Created/Modified files during execution:
# - output_image.jpg