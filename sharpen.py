import cv2
import numpy as np
# from deblurgan import DeblurGAN  

def sharpen(image, amount=1.0, sigma=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
def enhance_image(img, brightness=30, contrast=1.3, sharpen_amount=0.5):
    # Step 1: Brighten
    img_bilateral = cv2.bilateralFilter(img,  d=2, sigmaColor=1, sigmaSpace=1)
    denoised = cv2.fastNlMeansDenoisingColored(
                                                img,
                                                None,
                                                h=2,            # Start with lower luminance strength
                                                hColor=10,       # Lower color denoising
                                                templateWindowSize=5,
                                                searchWindowSize=15
                                            )
    # cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  

    brightened = cv2.convertScaleAbs(denoised, alpha=contrast, beta=brightness)

    # Step 2: Sharpen using unsharp masking
    gaussian = cv2.GaussianBlur(brightened, (0, 0), 2.0)
    sharpened = cv2.addWeighted(brightened, 1 + sharpen_amount,
                               gaussian, -sharpen_amount, 0)
    return sharpened  


# Usage
kernel = np.array([[-1,-1,-1],  
                   [-1, 9,-1],  
                   [-1,-1,-1]])  

# Apply the sharpening kernel  
img = cv2.imread('/media/hamit/HamitsKingston/processed_data/2025-05-20_18-21-50_tahir_2/ims_half_3/8/000003.png')
# result = cv2.filter2D(img, -1, kernel) 

# result = sharpen(img, amount=0.5)

# laplacian = cv2.Laplacian(img, cv2.CV_64F)

# # Convert back to uint8
# laplacian = np.uint8(np.absolute(laplacian))

# # Add to original image
# result = cv2.add(img, laplacian)

# result = cv2.convertScaleAbs(img, alpha=1.2, beta=30) 

# # For grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# equalized = cv2.equalizeHist(gray)

# # For color images - apply to each channel
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
# result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
result = enhance_image(img, brightness=0, contrast=1.4, sharpen_amount=0.5)


# # Convert to LAB color space
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# # Apply CLAHE to L channel
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
# lab[:,:,0] = clahe.apply(lab[:,:,0])

# # Convert back to BGR
# result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
# cv2.imwrite("orginial.png", img)
cv2.imwrite("sharpened.png", result)