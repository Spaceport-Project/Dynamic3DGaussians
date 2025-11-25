import json
import xml.etree.ElementTree as ET

import numpy as np
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

def parse_metashape_cameras(xml_path):
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
            # world_to_cam = np.array(cameras_from_opencv[int(cam_id)]).reshape(4,4)
            # cam_to_world = np.linalg.inv(world_to_cam)
            cameras[cam_id] = {"label": label, "pose": cam_to_world}
    return cameras, K, dist 
# Load from file
xml_path = "/DATA/processed_data/12-11-2025_Data/2025-11-12_14-20-42_ahmet_4_70_1870/cameras.xml"
with open('/home/hamit/Softwares/Dynamic3DGaussians/data/2025-11-12_14-20-42_ahmet_4_70_1870/train_meta.json', 'r') as file:
    data = json.load(file)
with open('/home/hamit/Softwares/Dynamic3DGaussians/data_making/data.json', 'r') as file:
    cam_poses= json.load(file)
for pose in cam_poses.keys():
    print(pose)
# camposes_arr = [np.linalg.inv(np.array(cam_poses[pose]["pose"])) for pose in cam_poses.keys()]
cams, K, dist =parse_metashape_cameras(xml_path)
for pose in cams.keys():
    print(pose)
# Now 'data' contains all components as a Python dict/list
camposes_arr = [np.array(cams[pose]["pose"]) for pose in cams.keys()]

data_new = {}
data_new["w"] = 3205
data_new["h"] = 2343
data_new['k'] = []
data_new["w2c"] =[]
data_new["fn"] = data["fn"]
data_new["cam_id"] = data["cam_id"]
K_list = [ K for _ in range(44)]
for dt in data["w2c"]:

    data_new["k"].append(K_list)

    data_new["w2c"].append((camposes_arr))



# print(repr(camposes_arr))
# print(data["w2c"])
# print("\n\n")
# print(data_new["w2c"])
# data_new["w2c"] = data_new["w2c"].tolist()
with open('train_meta.json', 'w') as f:  
    json.dump(convert_to_serializable(data_new), f, indent=4) 
