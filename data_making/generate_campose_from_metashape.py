import xml.etree.ElementTree as ET
import json

xml_path = "data_making/camposes_metashape.xml"  # your file

tree = ET.parse(xml_path)
root = tree.getroot()

poses = {}  # {camera_id: 4x4 list of lists}

# find the <chunk> and then <cameras>
for chunk in root.findall(".//chunk"):
    for cam in chunk.findall(".//camera"):
        cam_id = cam.attrib.get("id")
        transform_elem = cam.find("transform")
        if transform_elem is None or not transform_elem.text:
            continue

        # split string of 16 numbers into floats
        vals = [float(x) for x in transform_elem.text.split()]
        if len(vals) != 16:
            raise ValueError(f"Camera {cam_id}: expected 16 numbers, got {len(vals)}")

        # row-major 4x4
        mat4 = [
            vals[0:4],
            vals[4:8],
            vals[8:12],
            vals[12:16],
        ]

        poses[cam_id] = mat4

# Example: print pose for camera 0
print("Camera 0 world->camera pose:")
print(json.dumps(poses["0"], indent=4))

# Or dump all poses as a list of matrices (sorted by camera id)
all_poses_sorted = [poses[str(i)] for i in sorted(map(int, poses.keys()))]
print(json.dumps(all_poses_sorted, indent=4))