import json
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d

def parse_metashape_cameras(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cameras = {}
    for chunk in root.findall(".//chunk"):
        for cam in chunk.findall(".//camera"):
            cam_id = cam.attrib.get("id")
            label = cam.attrib.get("label", cam_id)
            tr = cam.find("transform")
            if tr is None or not tr.text:
                continue
            vals = [float(x) for x in tr.text.split()]
            if len(vals) != 16:
                continue
            cam_to_world = np.array(vals).reshape(4, 4)
            cameras[cam_id] = {"label": label, "pose": cam_to_world}
    return cameras

def parse_region_center(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for chunk in root.findall(".//chunk"):
        region = chunk.find(".//region")
        if region is not None:
            center_elem = region.find("center")
            if center_elem is not None and center_elem.text:
                vals = [float(x) for x in center_elem.text.split()]
                return np.array(vals)
    return None

def compute_scene_center_rays(cameras):
    """Compute scene center by ray intersection."""
    origins = []
    directions = []
    
    for cam_data in cameras.values():
        pose = cam_data["pose"]
        cam_pos = pose[:3, 3]
        origins.append(cam_pos)
        
        # Camera viewing direction (+Z axis in Metashape)
        view_dir = pose[:3, 2]
        directions.append(view_dir / np.linalg.norm(view_dir))
    
    origins = np.array(origins)
    directions = np.array(directions)
    
    A = []
    b = []
    for i in range(len(origins)):
        o = origins[i]
        d = directions[i]
        I_minus_ddT = np.eye(3) - np.outer(d, d)
        A.append(I_minus_ddT)
        b.append(I_minus_ddT @ o)
    
    A = np.vstack(A)
    b = np.hstack(b)
    center, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return center

def compute_rotation_align_vectors(v_from, v_to):
    """Compute rotation matrix that rotates v_from to v_to."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    
    if np.allclose(v_from, v_to):
        return np.eye(3)
    
    if np.allclose(v_from, -v_to):
        perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v_from, perp)
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    
    axis = np.cross(v_from, v_to)
    axis = axis / np.linalg.norm(axis)
    
    cos_angle = np.dot(v_from, v_to)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R

def align_and_recenter_cameras(cameras, scene_center, reference_cam_id="0"):
    """
    1. Align reference camera's up direction to negative Y
    2. Recenter so scene_center becomes [0, 0, 0]
    
    Returns:
        new_cameras: transformed camera poses
        new_center: new position of scene center (should be [0,0,0])
    """
    if reference_cam_id not in cameras:
        raise ValueError(f"Camera {reference_cam_id} not found")
    
    # Step 1: Align camera 0's up to negative Y
    ref_pose = cameras[reference_cam_id]["pose"]
    up_current = ref_pose[:3, 1]
    
    print(f"Camera {reference_cam_id} current up direction: {up_current}")
    
    up_target = np.array([0, 1, 0])
    R_align = compute_rotation_align_vectors(up_current, up_target)
    
    print(f"After alignment, up direction: {R_align @ up_current}")
    
    # Step 2: Apply rotation to all cameras
    T_align = np.eye(4)
    T_align[:3, :3] = R_align
    
    cameras_rotated = {}
    for cid, data in cameras.items():
        M_old = data["pose"]
        M_new = T_align @ M_old
        cameras_rotated[cid] = {
            "label": data["label"],
            "pose": M_new
        }
    
    # Step 3: Transform scene center by same rotation
    scene_center_homo = np.append(scene_center, 1)
    scene_center_rotated = (T_align @ scene_center_homo)[:3]
    
    print(f"Scene center after rotation: {scene_center_rotated}")
    
    # Step 4: Translate so scene center becomes origin
    T_recenter = np.eye(4)
    T_recenter[:3, 3] = -scene_center_rotated
    
    cameras_final = {}
    for cid, data in cameras_rotated.items():
        M_rotated = data["pose"]
        M_final = T_recenter @ M_rotated
        cameras_final[cid] = {
            "label": data["label"],
            "pose": M_final
        }
    
    # New scene center is now at origin
    new_center = np.array([0.0, 0.0, 0.0])
    # Verify: transform the scene center by the same operations
    scene_center_homo = np.append(scene_center_rotated, 1)
    new_center = (T_recenter @ scene_center_homo)[:3]

    print(f"Scene center after recentering (computed): {new_center}")
    print(f"Should be [0,0,0]: {np.allclose(new_center, [0, 0, 0])}")
    
    # print(f"Scene center after recentering: {new_center}")
    
    return cameras_final, new_center

def create_arrow_from_vector(origin, direction, length=0.5, color=[0, 1, 0]):
    """Create an arrow geometry pointing along 'direction' from 'origin'."""
    direction = np.asarray(direction, dtype=float)
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        raise ValueError("Direction vector is too small")
    direction = direction / direction_norm

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01,
        cone_radius=0.02,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2
    )
    arrow.paint_uniform_color(color)

    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, direction)
    c = np.dot(z, direction)

    if np.allclose(direction, z):
        R = np.eye(3)
    elif np.allclose(direction, -z):
        R = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])
    else:
        s = np.linalg.norm(v)
        vx = np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin

    arrow.transform(T)
    return arrow

def create_camera_frustum(pose, size=0.2, color=[1, 0, 0]):
    z = 2 * size
    x = size
    y = size

    points = np.array([
        [0, 0, 0],
        [-x, -y, z],
        [x, -y, z],
        [x, y, z],
        [-x, y, z],
    ])

    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_world = (pose @ pts_h.T).T[:, :3]

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls

def visualize_cameras_with_cam0_up_arrow(cameras, frustum_size=0.2, ref_point=None):
    geoms = []
    
    # World origin coordinate frame (should be at scene center now)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    if ref_point is not None:
        # Reference point sphere (should be at origin)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(ref_point)
        sphere.paint_uniform_color([1, 0, 0])
        geoms.append(sphere)

    colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
    ]

    # Add all camera frustums and centers
    for idx, (cid, data) in enumerate(sorted(cameras.items(), key=lambda x: int(x[0]))):
        pose = data["pose"]
        c = colors[idx % len(colors)]

        frustum = create_camera_frustum(pose, size=frustum_size, color=c)
        geoms.append(frustum)

        cam_pos = pose[:3, 3]
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        sph.translate(cam_pos)
        sph.paint_uniform_color(c)
        geoms.append(sph)

    # Arrow for camera 0 up direction
    cam0_pose = cameras["0"]["pose"]
    cam0_pos = cam0_pose[:3, 3]
    cam0_up = cam0_pose[:3, 1]   # Y-axis of camera 0 in world

    arrow = create_arrow_from_vector(
        origin=cam0_pos,
        direction=cam0_up,
        length=0.5,
        color=[0, 1, 0]  # green
    )
    geoms.append(arrow)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Cameras with Scene Center at Origin",
        width=1920,
        height=1080,
    )
def convert_to_serializable(obj):
    """Recursively convert numpy types to JSON serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj


if __name__ == "__main__":
    xml_path = "camposes_metashape.xml"

    # 1. Parse original cameras
    cams_old = parse_metashape_cameras(xml_path)
    print(f"Loaded {len(cams_old)} cameras")

    # 2. Get scene center
    scene_center = parse_region_center(xml_path)
    if scene_center is None:
        print("No region center found, computing from camera rays...")
        scene_center = compute_scene_center_rays(cams_old)
    
    print(f"Original scene center: {scene_center}")

    # 3. Align camera 0's up to negative Y and recenter scene
    cams_final, new_center = align_and_recenter_cameras(
        cams_old, 
        scene_center, 
        reference_cam_id="0"
    )

    # 4. Verify camera 0's up direction
    cam0_up = cams_final["0"]["pose"][:3, 1]
    print(f"Camera 0 final up direction: {cam0_up}")
    print(f"Is aligned to [0,-1,0]: {np.allclose(cam0_up, [0, -1, 0])}")
    print(cams_final)

    with open('data.json', 'w') as file:  
        json.dump(convert_to_serializable(cams_final), file, indent=4)  
    # 5. Visualize (reference point should be at origin now)
    print(repr(np.linalg.inv(cams_old["0"]["pose"])))
    visualize_cameras_with_cam0_up_arrow(cams_final, frustum_size=0.15, ref_point=new_center)