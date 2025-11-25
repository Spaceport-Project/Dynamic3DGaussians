import argparse
import xml.etree.ElementTree as ET
import json
import numpy as np
import cv2
import os

def undistort_images(image_directory, output_directory, camera_matrix, distortion_coefficients, img_ext):
    """Undistort images in the specified directory."""
    for file in os.listdir(image_directory):
        if not file.endswith(img_ext):
            continue
        image_path = os.path.join(image_directory, file)
        print('Processing image:', image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image {image_path}. Skipping.")
            continue

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)

        # mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, new_camera_matrix (w,h), 5)
        # undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]

        basename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_directory, basename), undistorted_image)


def create_cam_matrix_and_dist_coeff(xml_path):
    # ---- 1. Read Metashape XML from file ----
#    / xml_path = "camposes_metashape.xml"   # <-- change this to your filename

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Assuming one sensor (id=0) as in your snippet:
    calibration = root.find('.//sensor[@id="0"]/calibration')
    if calibration is None:
        raise RuntimeError("Could not find <calibration> node under sensor id=0")

    resolution = calibration.find('resolution')
    width = int(resolution.get('width'))
    height = int(resolution.get('height'))

    # ---- 2. Extract calibration parameters ----
    f  = float(calibration.find('f').text)
    cx = float(calibration.find('cx').text)
    cy = float(calibration.find('cy').text)
    k1 = float(calibration.find('k1').text)
    k2 = float(calibration.find('k2').text)
    k3 = float(calibration.find('k3').text)
    p1 = float(calibration.find('p1').text)
    p2 = float(calibration.find('p2').text)

    # Metashape cx, cy are offsets from image center → convert to absolute principal point
    pp_x = width  / 2.0 + cx
    pp_y = height / 2.0 + cy

    # ---- 3. Create camera matrix (NumPy) ----
    K = np.array([
        [f, 0,   pp_x],
        [0, f,   pp_y],
        [0, 0,   1.0 ]
    ], dtype=np.float64)

# ---- 4. Create distortion coefficients (NumPy, OpenCV convention) ----
# OpenCV: [k1, k2, p1, p2, k3]
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    return K, dist

# # ---- 5. Prepare JSON-serializable version ----
# calibration_data = {
#     "sensor_id": 0,
#     "image_resolution": {"width": width, "height": height},
#     "camera_matrix": K.tolist(),
#     "distortion_coefficients": dist.tolist(),
#     "calibration_parameters": {
#         "focal_length": f,
#         "principal_point": {"cx": pp_x, "cy": pp_y},
#         "radial_distortion": {"k1": k1, "k2": k2, "k3": k3},
#         "tangential_distortion": {"p1": p1, "p2": p2}
#     }
# }

# # ---- 6. Save to JSON ----
# with open("camera_calibration.json", "w") as f_out:
#     json.dump(calibration_data, f_out, indent=2)

# print("Camera matrix K:\n", K)
# print("Distortion coefficients (k1, k2, p1, p2, k3):\n", dist)
# print("Saved JSON -> camera_calibration.json")

def main():
    parser = argparse.ArgumentParser(description="Undistort images using camera parameters")
    parser.add_argument("image_directory", help="Directory containing images to undistort")
    parser.add_argument("output_directory", help="Directory to save undistorted images")
    parser.add_argument("--img_ext", default=".png", help="Image file extension to process (default: .png)")
    
    args = parser.parse_args()
    image_directory = os.path.join(args.image_directory)
    output_directory = os.path.join(args.output_directory, 'images')
    xml_file_path = os.path.join("./", 'camposes_metashape.xml')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("Loading images from", image_directory)
    print("Saving undistorted images to", output_directory)
    print("Loading camera parameters from", xml_file_path)

    camera_matrix, distortion_coefficients = create_cam_matrix_and_dist_coeff(xml_file_path)
    undistort_images(image_directory, output_directory, camera_matrix, distortion_coefficients, args.img_ext)

if __name__ == "__main__":
    main()