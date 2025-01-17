
# import xml.etree.ElementTree as ET

# # Parse the XML file
# tree = ET.parse('/home/hamit/Documents/agisoft_new.xml')
# root = tree.getroot()

# # Access the root element
# print(f"Root tag: {root.tag}")

# for child in root:
#     print(f"Tag: {child.tag}, Attributes: {child.attrib}, Text: {child.text}")

import numpy as np
import xmltodict

with open('/home/hamit/Documents/agisoft_new.xml') as file:
    # Parse XML to dictionary
    data_dict = xmltodict.parse(file.read())

# Access data like a regular dictionary
width = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['resolution']["@width"])
height = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['resolution']["@height"])
fx = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['f'])
fy = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['f'])
cx = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['cx']) + (width - 1)/2
cy = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['cy']) + (height - 1)/2
k1 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['k1'])
k2 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['k2'])
k3 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['k3'])
p1 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['p1'])
p2 = float(data_dict['document']['chunk']['sensors']['sensor']['calibration']['p2'])

camera_matrix = K = np.array([
                [fx , 0, cx ],
                [0, fy, cy],
                [0, 0, 1]
            ])
distortion_coffs = np.array([k1, k2, p1, p2, k3])


len (data_dict['document']['chunk']['cameras']['camera'])
cam_pose_list = []
for id, cam in enumerate(data_dict['document']['chunk']['cameras']['camera']):
    print(cam['@id'])
    transform = [float(t)  for t in cam['transform'].split() ]
    cam_pose_list.append(np.array(transform).reshape(4,4))


