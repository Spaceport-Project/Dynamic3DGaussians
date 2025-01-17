import os
import json
import argparse
import sys
import copy

intrincs =  [[
                [
                    465.6666666666667,
                    0.0,
                    318.3373333333333
                ],
                [
                    0.0,
                    464.43333333333334,
                    186.84466666666668
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    466.3833333333333,
                    0.0,
                    313.69899999999996
                ],
                [
                    0.0,
                    465.3966666666667,
                    185.387
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    466.05333333333334,
                    0.0,
                    316.9036666666667
                ],
                [
                    0.0,
                    464.78333333333336,
                    188.36566666666664
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    544.7733333333333,
                    0.0,
                    316.4
                ],
                [
                    0.0,
                    543.51,
                    190.52866666666668
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    465.61333333333334,
                    0.0,
                    311.04866666666663
                ],
                [
                    0.0,
                    464.45333333333326,
                    186.65099999999998
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    465.2733333333333,
                    0.0,
                    317.1836666666667
                ],
                [
                    0.0,
                    464.06666666666666,
                    182.33633333333333
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    465.65,
                    0.0,
                    316.3976666666667
                ],
                [
                    0.0,
                    464.5933333333333,
                    182.98333333333332
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    547.4133333333334,
                    0.0,
                    315.6226666666667
                ],
                [
                    0.0,
                    546.03,
                    186.93766666666667
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    466.17,
                    0.0,
                    315.695
                ],
                [
                    0.0,
                    464.99666666666667,
                    180.4736666666667
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    497.6166666666667,
                    0.0,
                    311.8550000000001
                ],
                [
                    0.0,
                    496.43,
                    182.20333333333335
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    465.38000000000005,
                    0.0,
                    322.17966666666666
                ],
                [
                    0.0,
                    464.1933333333333,
                    187.63966666666664
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    530.2966666666667,
                    0.0,
                    313.1526666666667
                ],
                [
                    0.0,
                    529.0966666666667,
                    186.92066666666662
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    549.87,
                    0.0,
                    312.0946666666667
                ],
                [
                    0.0,
                    548.3666666666667,
                    188.99733333333333
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    476.04333333333335,
                    0.0,
                    316.81766666666664
                ],
                [
                    0.0,
                    474.71333333333337,
                    182.39966666666666
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    464.4866666666667,
                    0.0,
                    305.6449999999999
                ],
                [
                    0.0,
                    463.5199999999999,
                    187.18466666666666
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    514.6,
                    0.0,
                    316.3256666666667
                ],
                [
                    0.0,
                    513.2733333333333,
                    184.42900000000003
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    465.8133333333334,
                    0.0,
                    317.0756666666667
                ],
                [
                    0.0,
                    464.6666666666667,
                    184.60033333333334
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    467.69666666666666,
                    0.0,
                    313.6453333333333
                ],
                [
                    0.0,
                    466.5066666666667,
                    184.29633333333334
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    465.93333333333334,
                    0.0,
                    310.865
                ],
                [
                    0.0,
                    464.65999999999997,
                    187.32766666666666
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    505.5233333333333,
                    0.0,
                    315.37033333333335
                ],
                [
                    0.0,
                    504.20666666666665,
                    186.536
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    524.3966666666668,
                    0.0,
                    314.37666666666667
                ],
                [
                    0.0,
                    522.82,
                    186.30133333333333
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    466.51,
                    0.0,
                    318.372
                ],
                [
                    0.0,
                    465.29333333333335,
                    181.55900000000003
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    465.9,
                    0.0,
                    311.748
                ],
                [
                    0.0,
                    464.8833333333333,
                    186.00300000000001
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    539.01,
                    0.0,
                    317.51933333333335
                ],
                [
                    0.0,
                    537.6166666666667,
                    183.84233333333336
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    620.56,
                    0.0,
                    312.3056666666667
                ],
                [
                    0.0,
                    618.9733333333334,
                    183.359
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    465.31000000000006,
                    0.0,
                    317.684
                ],
                [
                    0.0,
                    464.2300000000001,
                    186.97200000000004
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    466.8433333333333,
                    0.0,
                    313.575
                ],
                [
                    0.0,
                    465.7366666666667,
                    190.48733333333334
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
]




def get_intr_and_camera_poses(sfm_file):
    view_id = dict()
    c2w = []
    ks = []
    with open(sfm_file, 'r') as f:
        data=json.load(f)
        data_intr = []
        # intr_0 = data['intrinsics'][0]
        # w = int(intr_0['width'])
        # h = int(intr_0['height'])
        # sensor_width = int(intr_0['sensorWidth'])
        # sensor_height = int(intr_0['sensorHeight'])
        # focal_length = float(intr_0['focalLength'])
        # fx = max(w,h ) * focal_length / max(sensor_width, sensor_height)
        # fy = fx
        # cx = w/2
        # cy = h/2
        # k = [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
        intrincs_id = int(data['intrinsics'][0]['intrinsicId'])
        f_out = open("viewpoints.sfm","w")
        for k, intr in enumerate(intrincs):
            [ [fx, _, cx], [_, fy, cy], [_, _, _] ] = intr
            intr_clone = copy.deepcopy(data['intrinsics'][0])
            intr_clone['focalLength'] =str( (fx * int(intr_clone['sensorWidth'])) / int(intr_clone['width']))
            intr_clone['pixelRatio'] = str(fx/fy)
            intr_clone['principalPoint'][0] = str(cx - int(intr_clone['width'])/2)
            intr_clone['principalPoint'][1] = str(cy - int(intr_clone['height'])/2)
            if k != 0:
                intrincs_id += 1
                intr_clone['intrinsicId'] = str(intrincs_id)

            data_intr.append(intr_clone)
       

        for item in data['views']:
            # print (item)
            view_id[item['path'].split('/')[-1].split('.')[0]] = item['viewId'] 
        
        cam_indices = list(sorted(view_id.keys(), key=lambda x:int(x)))

        for k, ind in enumerate(cam_indices):
            for view in data["views"]:
                if view['viewId'] ==  view_id[ind]:
                    view["intrinsicId"] = data_intr[k]["intrinsicId"]
                    
        data['intrinsics'] = data_intr
        json.dump(data, f_out, indent=4)
        f_out.close()

        


def main():
    parser = argparse.ArgumentParser(description="Create camera init sfm for meshroom")
    parser.add_argument("--input_sfm_file", required=True, help="This is the path to the folder containing the images, and where train_meta.json and init_pt_cld.npz will be written. In the ims folder, each subfolder is a camera")

    args = parser.parse_args()
    get_intr_and_camera_poses(args.input_sfm_file)
    sys.exit(0)

if __name__ == "__main__":
    main()