import copy
import open3d as o3d
import numpy as np
def create_rotation_matrix(look_at, up):
    # Normalize the look-at vector (this will be our forward vector)
    forward = look_at / np.linalg.norm(look_at)

    # Calculate right vector using cross product of forward and up
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    # Recalculate the orthogonal up vector
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    # Create the rotation matrix
    # Each row represents one of our basis vectors
    rotation_matrix = np.array([
        [right[0],   right[1],   right[2]],
        [up[0],      up[1],      up[2]],
        [forward[0], forward[1], forward[2]]
    ])

    return rotation_matrix
def compute_rotation_matrix(look_at, up):
    # Normalize the look-at vector
    z = look_at / np.linalg.norm(look_at)
    
    # Compute the right vector (cross product of up and look-at)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    
    # Compute the true up vector (cross product of look-at and right)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    
    # Construct the rotation matrix (column-major)
    rotation_matrix = np.array([
        [x[0], y[0], z[0]],
        [x[1], y[1], z[1]],
        [x[2], y[2], z[2]]
    ])
    
    return rotation_matrix
if __name__ == '__main__':

    # w2c = np.array([   [ 0.25681035,  0.0670684,  -0.96413188, 0.65729515 ],
    #     [-0.38885296,  0.92045067, -0.03954678, -0.9941085],
    #     [ 0.88478349,  0.38506155,  0.262461, 4.50231357 ],
    #     [0,0,0,1]
    # ] )
    # w2c = np.array([[ 0.93674253, -0.03888465,  0.34785257, -0.20542087],
    #    [ 0.16813808,  0.92162764, -0.34976003, -0.70957294],
    #    [-0.30699025,  0.38612236,  0.8698658 ,  3.68601693],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    Up = np.array([0.262550099212965,0.820548910646681,0.507707524702765])
    LookAt = np.array([-1.16875486397154,0.82118750743449,-3.09825925371493 ])
    # rotation = compute_rotation_matrix(LookAt, Up)
    # rotation = np.array([[ -6.1258866741e-01, -2.6478619283e-01, 7.4473041877e-01],
    #         [2.6255009921e-01, 8.2054891065e-01, 5.0770752470e-01],
    #         [-7.4552167639e-01, 5.0654492133e-01, -4.3313932252e-01]])
    
    # translation = np.array([1.8088410405e+00, 1.2060417274e+00, -3.6292783610e+00])

    rotation =   np.array([[-0.61258866740844165, -0.26478619283153432, 0.74473041877482848],
              [-0.26255009921296457, -0.82054891064668067, -0.50770752470276459],
              [0.74552167638909916,  -0.50654492132714857, 0.43313932251835618]])
           
    center = np.array([-1.9142765403606425 , 1.3277324287616383, -3.5313985762332871])
    # c2w = np.vstack((np.concatenate((rotation, center[:,None]), axis=1),[0,0,0,1]))
    c2w = np.array([-0.61258866740844176, -0.26255009921296429, 0.74552167638909927, -1.9142765403606425,
                     -0.26478619283153432, -0.82054891064668067, -0.50654492132714823, 1.3277324287616383,
                       0.7447304187748286, -0.50770752470276426, 0.43313932251835613, -3.5313985762332871, 
                       0, 0, 0, 1]).reshape(4,4)
    # c2w = np.array([0.61258866740844176, -0.26255009921296429, 0.74552167638909927, -1.9142765403606425,
    #                 0.26478619283153432, -0.82054891064668067, -0.50654492132714823, 1.3277324287616383,
    #                     -0.7447304187748286, 0.50770752470276426, 0.43313932251835613, -3.5313985762332871,
    #                     0, 0, 0, 1]).reshape(-1,4)
    w2c = np.linalg.inv(c2w)
    path = "/home/hamit/DATA/19-12-2024_Data/2024-12-19_19-12-14_4096/processed_data_300-600_calib/points3D.ply"
    
    point_cloud = o3d.io.read_point_cloud(path)
    print(point_cloud.get_center())
    point_cloud_trans = copy.deepcopy(point_cloud).transform(w2c)
    o3d.io.write_point_cloud("/home/hamit/DATA/19-12-2024_Data/2024-12-19_19-12-14_4096/processed_data_300-600_calib/points3D_agisoft_trans.ply", point_cloud_trans)
    # o3d.visualization.draw_geometries([point_cloud, point_cloud_trans])
