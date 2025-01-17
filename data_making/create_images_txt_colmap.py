import cv2
import numpy as np


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
def load_cameras_params(calib_cameras_data_path):
    """Load camera parameters from a JSON file."""
    fs = cv2.FileStorage(str(calib_cameras_data_path), cv2.FILE_STORAGE_READ)
    num_cameras =  int(fs.getNode("nb_camera").real())
    camera_matrix_list = []
    distortion_coefficients_list = []

    w2c_list = []
    for cam_idx in range(num_cameras):
        cam_name = f"camera_{cam_idx}"
        camera_matrix = fs.getNode(cam_name).getNode("camera_matrix").mat()
        # with np.nditer(camera_matrix, op_flags=['readwrite']) as it:
        #     for x in it:
        #         if x != 0 and x !=1 :
        #             x *= 2 
        # camera_matrix_ = [it if (it == 1 or it == 0) else it/2  for it in camera_matrix ]
        camera_matrix_list.append(camera_matrix)
        distortion_coefficients = fs.getNode(cam_name).getNode("distortion_vector").mat()
        distortion_coefficients_list.append(distortion_coefficients)
        
        camera_pose_matrix = fs.getNode(cam_name).getNode("camera_pose_matrix").mat()
       

        # camera_pose_matrix[:3,3] *= scale/1000
        # camera_pose_matrix = c2w_0 @ camera_pose_matrix
        w2c = np.linalg.inv(camera_pose_matrix)
        # w2c[:3,3] *= 15.86/1000
        w2c_list.append(w2c.tolist())
    return camera_matrix_list, distortion_coefficients_list, w2c_list
