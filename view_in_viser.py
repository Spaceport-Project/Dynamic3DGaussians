
import torch
import numpy as np
import open3d as o3d
import time
import os
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult, searchForMaxIteration
from external import build_rotation
from colormap import colormap
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
# import ffmpeg
import sys
from PIL import Image
import gc
import viser
from viser import transforms as tf
import imageio.v3 as iio
import signal


REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

w, h = 1920, 1080
near, far = 0.01, 100.0

def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()
cnt=0

to8b = lambda x : (255*np.clip(x.permute(1,2,0).cpu().detach().numpy(),0,1)).astype(np.uint8)




def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )




  
        
class Viewer():
    # roll_limit=((0.4, 0.4 - 3.14), (-1.1, -1.1 + 3.14))
    roll_limit=((0.4, -1.1), (3.14 - 1.1, 0.4 - 3.14))
    pitch_limit=((1.3, -1.5), (3.14 - 1.5, 1.3 - 3.14))

    # S03_roll_limit = tf.SO3.from_x_radians(roll_limit[0]).as_matrix(), tf.SO3.from_x_radians(roll_limit[1]).as_matrix()
    def __init__(self, seq, exp, f_ratio=0.8, w=1920, h=1080, near=0.01, far=100.0):
        self.seq = seq
        self.exp = exp
        self.viser_server = viser.ViserServer(port=8081)
        self.viser_server.scene.world_axes.visible = True
        self.viser_server.on_client_connect(self.handle_new_client)
        self.viser_server.on_client_disconnect(self.handle_disconnect_client)
        self.clients_num = 0
        self.k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.cnt = dict()
        

    def handle_disconnect_client(self, client:viser.ClientHandle):
        self.cnt[client.client_id] = False
        print(f"{client.client_id} client disconnected!")


    
    def handle_new_client(self, client:viser.ClientHandle):
        print("new client!")
        # This will run whenever we get a new camera!
        # @client.camera.on_update
        # def _(_: viser.CameraHandle) -> None:
        #     self.cnt[self.clients_num]=0
        #     print(f"New camera on client {client.client_id}!")
            
        self.clients_num +=1 
        # Show the client ID in the GUI.
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = False
        print(client.client_id)
        self.cnt[client.client_id]=0
    
    
    def S03_roll_limit(self, pitch, yaw):
        return tf.SO3.from_rpy_radians(self.roll_limit[0], pitch, yaw), tf.SO3.from_rpy_radians(self.roll_limit[1], pitch, yaw)
    def S03_pitch_limit(pitch):
        return tf.SO3.from_y_radians(pitch).as_matrix()
    def S03_yaw_limit(yaw):
        return tf.SO3.from_y_radians(yaw).as_matrix()

    def render_images(self):
        
        # params = self.load_params_data(self.seq, self.exp)
        # num_timesteps = len(params['means3D'])
        cnt=0
        scene_data, _ = self.load_scene_data2(self.seq, self.exp, seg_as_col=False)
        num_timestamps = len(scene_data)
        # clients = self.viser_server.get_clients()
        # for id1, _ in clients.items():
        #     self.cnt[id1]=0
        c2w=np.eye(4)
        flag = False
        prev_flag = False
        while True:

            # for step in range(num_timesteps//limit_timestep + 1):
            #     if  step == num_timesteps//limit_timestep:
            #         low_up_limit = (limit_timestep*step, num_timesteps)
            #     else:
            #         low_up_limit = (limit_timestep*step, limit_timestep*(step+1))
                
                # scene_data, _ = self.load_scene_data(params, low_up_limit, seg_as_col=False)

                
                # Get all currently connected clients.
                # num_timestamps = len(scene_data)
                for t in range(num_timestamps):   
                    
                    clients = self.viser_server.get_clients()

                
                    for id, client in clients.items():
                        # if cnt == 0:
                        #     client.camera.wxyz = self.init_camera()[0]
                        #     client.camera.position = self.init_camera()[1]
                        if  self.cnt[client.client_id] == 0:
                            
                            client.camera.wxyz = self.init_camera()[0]
                            client.camera.position = self.init_camera()[1] 
                            # client.camera.look_at= np.array([0, 1, 3.5]) #for oguz
                            client.camera.look_at= np.array([1, 1, 3.5]) # for yoga 

                            print(client.camera.look_at)
                            # print(client.camera.wxyz, client.camera.position)
                            # self.cnt[client.client_id] = True
                            
                        R_S03 = tf.SO3(np.asarray(client.camera.wxyz))
                        R_S03_inv = tf.SO3(np.asarray(client.camera.wxyz)).inverse()
                        R = R_S03.as_matrix()
                        T = client.camera.position

                        # print(R_S03.compute_roll_radians())
                        # if (R_S03.compute_roll_radians() >  self.roll_limit[0][0]  and R_S03.compute_roll_radians() >  self.roll_limit[0][1] ) or \
                        #         (R_S03.compute_roll_radians() <  self.roll_limit[1][0]  and R_S03.compute_roll_radians() <  self.roll_limit[1][1] ) :
                        print(self.roll_limit[0][0], self.roll_limit[0][1])
                        if (R_S03.compute_roll_radians() <  self.roll_limit[0][0]  and R_S03.compute_roll_radians() >  self.roll_limit[0][1] ) \
                             and (R_S03.compute_pitch_radians() <  self.pitch_limit[0][0]  and R_S03.compute_pitch_radians() >  self.pitch_limit[0][1]) :
                              # or (R_S03.compute_roll_radians() <  self.roll_limit[1][0]  and R_S03.compute_roll_radians() >  self.roll_limit[1][1] ) :
                        #if True:
                            if not flag:
                                T_prev = T.copy()
                                flag = True

                            # R  = self.S03_roll_limit(R_S03.compute_pitch_radians(), R_S03.compute_yaw_radians())[1]
                            # print("inside:", R_S03.as_rpy_radians())

                            # client.camera.wxyz = R.wxyz
                            # R = R.as_matrix() 
                            # c2w[:3,:3] = R
                            # c2w[:3,3] = T_prev
                            # T = T_prev
                            # w2c = np.linalg.inv(c2w)
                            # R_w2c = tf.SO3.from_matrix(w2c[:3,:3])
                            flag = False
                            # print(R_S03.as_rpy_radians())

                            c2w[:3,:3] = R
                            c2w[:3,3] = T
                            # T_prev = T.copy()
                            # c2w = np.vstack((np.concatenate((R, T[:,None]), axis=1),[0,0,0,1]))
                            w2c = np.linalg.inv(c2w)
                            print(tf.SO3.from_matrix(c2w[:3,:3]).as_rpy_radians())

                            w2c_prev = w2c.copy()
                            # if id == 0 and flag : 
                            #     flag=False
                            #     start = time.time()
                            start = time.time()



                           
                         
                        else:
                            
                            w2c = w2c_prev
                            c2w = np.linalg.inv(w2c_prev)

                            client.camera.position = c2w[:3,3] 
                            client.camera.wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
                            client.camera.look_at = np.array([1, 1, 3.5]) 
                            print("inside", tf.SO3.from_matrix(c2w[:3,:3]).as_rpy_radians())

                        # print(c2w, w2c)
                        # print(R, T)

                        print(client.camera.look_at)
                        img, _ = self._render(w2c, scene_data[t], bg=[0, 0, 0])
                      
                       
                        img_num = to8b(img)
                        # self.viser_server.scene.set_background_image(
                        #     img_num,
                        #     format="png"  
                        # )

                        client.scene.set_background_image(
                            img_num,
                            format="jpeg",
                            jpeg_quality=80
                        )
                        # if id == 1 and not flag:
                        #     flag = True
                        #     end = time.time()
                        #     print("Render fps:", 1/(end-start))
                        end = time.time()
                        # print("Render fps:", 1/(end-start))

                        
                        # print(f"Camera pose for client {id}")
                        # print(f"\twxyz: {client.camera.wxyz}")
                        # print(f"\tc2w: {tf.SO3(np.asarray(client.camera.wxyz)).as_matrix()}")
                        # print(f"\tposition: {client.camera.position}")
                        # print(f"\tfov: {client.camera.fov}")
                        # print(f"\taspect: {client.camera.aspect}")
                        # print(f"\tlast update: {client.camera.update_timestamp}")
                        
                        # cnt += 1
                        
                        self.cnt[client.client_id] += 1
                    # time.sleep(0.02)


    def load_scene_data(self, params, low_upper_limit, seg_as_col=False):
        
        
        is_fg = params['seg_colors'][:, 0] > 0.5
        scene_data = []
        for t in range(*low_upper_limit):
            
            rendervar = {
                'means3D': params['means3D'][t].cuda(),
                'colors_precomp': params['rgb_colors'][t].cuda() if not seg_as_col else params['seg_colors'].cuda(),
                'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t].cuda()),
                'opacities': torch.sigmoid(params['logit_opacities']).cuda(),
                'scales': torch.exp(params['log_scales']).cuda(),
                'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
            }
            # rendervar = {k: v.cuda() for k, v in rendervar.items()}
            if REMOVE_BACKGROUND:
                rendervar = {k: v[is_fg] for k, v in rendervar.items()}
            scene_data.append(rendervar)
        if REMOVE_BACKGROUND:
            is_fg = is_fg[is_fg]
        return scene_data, is_fg
    
    def load_scene_data2(self, seq, exp, seg_as_col=False):
    
      
        params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    

        params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
        is_fg = params['seg_colors'][:, 0] > 0.5
        scene_data = []
        for t in range(len(params['means3D'])):
            rendervar = {
                'means3D': params['means3D'][t],
                'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
                'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
                'opacities': torch.sigmoid(params['logit_opacities']),
                'scales': torch.exp(params['log_scales']),
                'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
            }
            if REMOVE_BACKGROUND:
                rendervar = {k: v[is_fg] for k, v in rendervar.items()}
            scene_data.append(rendervar)
        if REMOVE_BACKGROUND:
            is_fg = is_fg[is_fg]
        return scene_data, is_fg

    
    
    def load_params_data(self, seq, exp):
        print(f"Loading params.npz file")
        params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
        params = {k: torch.tensor(v).float() for k, v in params.items()}
        print(f"Loaded params.npz file")

        return params
    
    def _render(self, w2c, timestep_data, bg=[0,0,0]):
        with torch.no_grad():
            cam = setup_camera(self.w, self.h, self.k, w2c, self.near, self.far, bg=torch.tensor(bg)) #[0, 177./255, 64.0/255]
            im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
            # im[~is_fg] =  torch.tensor([0, 177./255, 64.0/255], dtype=torch.float32, device="cuda")
            # torchvision.utils.save_image(im, '{0:05d}'.format(cnt) + ".png")

            return im, depth
    @classmethod
    def init_camera(cls, y_angle=0., center_dist=5.5, cam_height= -1, f_ratio=0.82):
        ry = y_angle * np.pi / 180
        # w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -0.0],
        #                 [0.,         1., 0.,          cam_height],
        #                 [np.sin(ry), 0., np.cos(ry),  center_dist],
        #                 [0.,         0., 0.,          1.]])
        w2c = np.array([[np.cos(ry), 0., -np.sin(ry), -2.0],
                        [0.,         1., 0.,          cam_height],
                        [np.sin(ry), 0., np.cos(ry),  center_dist],
                        [0.,         0., 0.,          1.]])
        c2w = np.linalg.inv(w2c)
        wxyz = tf.SO3.from_matrix(c2w[:3,:3]).wxyz
        return wxyz, c2w[:3,3]  
        


# def signal_handler(sig, frame):

#     print('You pressed Ctrl+C!')
#     for i in range(viewer.clients_num):
#         viewer.cnt[i]=0
#     viewer.viser_server.stop()
#     sys.exit(0)

if __name__ == "__main__":
    # signal.signal(signal.SIGINT, signal_handler)

    limit_timestep = 100

    # exp_name = "exp_black_onlyoguz_scl_2_full"
    # for sequence in ["oguz_2"]:

    # exp_name = "exp_only_oguz_2_scl_4_it_500_green_test2"
    # exp_name = "exp_witback_oguz_2_scl_4_it_500_green_test"
    
    

    # exp_name = "exp_withbck_test"
    # exp_name = "exp_withbck_scl_2_reduced"
    
    # exp_name = "exp_withbck_scl_2_it_500"
    # sequence = "10-09-2024_data/pose_1_3"
    
    # exp_name = "yoga_wo_background_onlypt_scale_4_it_500_pose_1_4_all"
    # sequence = "10-09-2024_data/pose_1_4_all"


    exp_name = "exp_only_oguz_2_scl_4_it_500_20cams"
    sequence = "oguz_2"

        
    viewer = Viewer(seq=sequence, exp=exp_name,w=1700, h=956)
    time.sleep(0.2)
    viewer.render_images()

