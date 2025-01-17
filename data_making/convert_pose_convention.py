from enum import IntEnum
from typing import Dict



def read_camera_poses_from_file(file_path, scale=1):
    # Initialize an empty list to store the camera poses
    camera_poses = []
    # Open the file and read all lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Process every set of 4 lines at a time since each camera pose is represented by a 4x4 matrix
        for i in range(0, len(lines), 4):  # Adjusted to 5 assuming there's a blank line between matrices
            # Extract four lines and create a 4x4 matrix
            pose_lines = lines[i:i+4]
            pose_matrix = np.array([[float(value) for value in line.strip().split()] for line in pose_lines])
            # Append the constructed matrix to the camera_poses list
            # inv_mat = np.linalg.inv(pose_matrix)

            R = pose_matrix[:3,:3]
            T = pose_matrix[:3,3]
            # T[1:] = -T[1:]
            # # R = -R
            # # R[:,0] = -R[:,0]
            mat  = np.eye(4)
            # R[1:, 0] *= -1
            # R[0, 1:] *= -1
            # # R=R.T
            # T = -np.matmul(R , T)
            mat[:3,:3] = R
            mat[:3, 3] = T*scale    


            # inv_mat = np.linalg.inv(mat)
            camera_poses.append(mat)

    return camera_poses
def get_camera_vertex(data):

    tranlations = []
    rotations = []
    transformations = []
    for d in data:
        # camera_info = {
        #     'translation': d[:3, 3],
        #     'rotation': d[:3, :3],
        #     'height': 1920,
        #     'width': 1200,
        #     'focal_length': 1361,
        # }
        
        rotations.append(d[:3,:3])
        tranlations.append(d[:3,3])
        transformations.append(d[:4,:4])
    return  np.array(transformations)

class CubeFace(IntEnum):
    FRONT = 0
    RIGHT = 1
    BACK = 2
    LEFT = 3
    UP = 4
    DOWN = 5


OPPOSITE_CUBEFACES: Dict[CubeFace, CubeFace] = {
    CubeFace.FRONT: CubeFace.BACK,
    CubeFace.RIGHT: CubeFace.LEFT,
    CubeFace.BACK: CubeFace.FRONT,
    CubeFace.LEFT: CubeFace.RIGHT,
    CubeFace.UP: CubeFace.DOWN,
    CubeFace.DOWN: CubeFace.UP
}

from typing import Tuple


def split_axes_convention(convention: str) -> Tuple[CubeFace, CubeFace, CubeFace]:
    """Parse the three axes contained in the convention string, e.g. LEFT_UP_FRONT"""
    splits = convention.upper().split("_")
    assert len(splits) == 3
    return CubeFace[splits[0]], CubeFace[splits[1]], CubeFace[splits[2]]

from typing import Dict, Final, Tuple


def get_reference_axes(ref_convention: str) -> Dict[CubeFace,
                                                    Tuple[float, float, float]]:
    """Get the reference coordinates of the cube directions"""
    x, y, z = split_axes_convention(ref_convention)
    return {
        x: (1., 0., 0.),
        y: (0., 1., 0.),
        z: (0., 0., 1.),
        OPPOSITE_CUBEFACES[x]: (-1., 0., 0.),
        OPPOSITE_CUBEFACES[y]: (0., -1., 0.),
        OPPOSITE_CUBEFACES[z]: (0., 0., -1.)
    }


REF_CONVENTION: Final[str] = "LEFT_UP_FRONT"
REF_AXES: Final[Dict[CubeFace,
                     Tuple[float, float, float]]] = get_reference_axes(REF_CONVENTION)

import numpy as np


def get_transform_to_ref(convention: str) -> np.ndarray:
    """Get the 3x3 matrix mapping from the input convention to the reference one"""
    x, y, z = split_axes_convention(convention)
    return np.column_stack((REF_AXES[x], REF_AXES[y], REF_AXES[z]))

from enum import IntEnum
from typing import Dict, Final

import numpy as np


class CoordinateSystem(IntEnum):
    REFERENCE = 0
    PYTORCH_3D = 1
    OPENCV = 2
    COLMAP = 3
    OPENGL = 4
    NGP = 5


CAMERA_TO_REF: Final[Dict[CoordinateSystem, np.ndarray]] = {
    CoordinateSystem.REFERENCE: get_transform_to_ref(REF_CONVENTION),
    CoordinateSystem.PYTORCH_3D: get_transform_to_ref("LEFT_UP_FRONT"),
    CoordinateSystem.COLMAP: get_transform_to_ref("RIGHT_DOWN_FRONT"),
    CoordinateSystem.OPENCV: get_transform_to_ref("RIGHT_DOWN_FRONT"),
    CoordinateSystem.OPENGL: get_transform_to_ref("RIGHT_UP_BACK"),
    CoordinateSystem.NGP: get_transform_to_ref("RIGHT_UP_BACK")
}


WORLD_TO_REF: Final[Dict[CoordinateSystem, np.ndarray]] = {
    CoordinateSystem.NGP: get_transform_to_ref("FRONT_LEFT_UP")
}

from typing import Tuple
import numpy as np


def convert_vertices(verts_in: np.ndarray,
                     system_in: CoordinateSystem,
                     system_out: CoordinateSystem) -> np.ndarray:
    """Transform vertices of shape (N,3) between different coordinate systems."""
    T_ref_win = WORLD_TO_REF.get(system_in, CAMERA_TO_REF[system_in])
    T_ref_wout = WORLD_TO_REF.get(system_out, CAMERA_TO_REF[system_out])
    T_wout_ref = np.linalg.inv(T_ref_wout)

    verts_out = np.linalg.multi_dot((T_wout_ref, T_ref_win, verts_in.T)).T
    return verts_out


def convert_pose(r_in: np.ndarray,
                 t_in: np.ndarray,
                 system_in: CoordinateSystem,
                 system_out: CoordinateSystem) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pose from one coordinate system to another."""
    T_ref_cin = CAMERA_TO_REF[system_in]
    T_ref_win = WORLD_TO_REF.get(system_in, T_ref_cin)
    T_cin_ref = np.linalg.inv(T_ref_cin)

    T_ref_cout = CAMERA_TO_REF[system_out]
    T_ref_wout = WORLD_TO_REF.get(system_out, T_ref_cout)
    T_wout_ref = np.linalg.inv(T_ref_wout)

    r_out = np.linalg.multi_dot((T_wout_ref, T_ref_win, r_in, T_cin_ref, T_ref_cout))
    t_out = np.linalg.multi_dot((T_wout_ref, T_ref_win, t_in))
    return r_out, t_out
import open3d as o3d
if __name__ == '__main__':
    camera_poses_data = read_camera_poses_from_file("/home/hamit/Softwares/spaceport-tools/data/cam_poses_meshroom_juggle2.txt",scale=2)
    camera_poses_data2 = read_camera_poses_from_file("/home/hamit/Softwares/spaceport-tools/data/cam_poses_train_2.txt")
    # camera_poses_data2 = read_camera_poses_from_file("/home/hamit/Softwares/spaceport-tools/data/cam_poses_colmap2.txt")

    # camera_poses = get_camera_vertex(camera_poses_data) 
    camera_poses_new = []
    print(len(camera_poses_data), len(camera_poses_data2))
    # for idx, cam in enumerate(camera_poses_data):
    #     camera_poses_data[idx][:3,:3], camera_poses_data[idx][:3,3] = convert_pose(cam[:3,:3], cam[:3,3], CoordinateSystem.OPENCV, CoordinateSystem.REFERENCE) 
    
    # convert_pose(camera_poses_data[0][:3,:3], camera_poses_data[0][:3,3], CoordinateSystem.OPENCV, CoordinateSystem.OPENGL)   
    camera_poses = get_camera_vertex(camera_poses_data) 
    camera_poses2 = get_camera_vertex(camera_poses_data2) 
   
    # file.close()

    points = [
    [0, 0, 0],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2]
    ]
    lines = [
    [0, 1],
    [0, 2],
    [0, 3],
    ]
    # points[2] = np.matmul(np.linalg.inv(rot0), points[2])
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    geometries =[ line_set]
    for geometry in geometries:
        viewer.add_geometry(geometry)
    # for k, cam in enumerate(render_poses):
    #     frustum = o3d.geometry.LineSet.create_camera_visualization(
    #         640, 480, intrinsic.intrinsic_matrix,
    #         np.linalg.inv(cam), 0.2)
    #     if k == 0:
    #         frustum.paint_uniform_color([0, 0, 1.000])
    #     else:
    #         frustum.paint_uniform_color([0.961, 0.475, 0.000])
    #     viewer.add_geometry(frustum)
    for k, cam in enumerate(camera_poses):
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            640, 480, intrinsic.intrinsic_matrix,
            cam, 0.2)
        if k == 0:
            frustum.paint_uniform_color([0, 0, 1.000])
        else:
            frustum.paint_uniform_color([0.961, 0.475, 0.000])
        viewer.add_geometry(frustum)
    for k, cam in enumerate(camera_poses2):
   
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            640, 480, intrinsic.intrinsic_matrix,
            cam, 0.2)
        if k == 0:
            frustum.paint_uniform_color([0, 0, 1.000])
        else:
            frustum.paint_uniform_color([0, 1, 0.000])
        viewer.add_geometry(frustum)
  
  
    opt = viewer.get_render_option()
    
    viewer.run()
    viewer.destroy_window()