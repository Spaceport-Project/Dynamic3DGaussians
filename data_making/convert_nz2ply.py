import numpy as np
from plyfile import PlyElement, PlyData # Requires plyfile==0.8.1


def construct_list_of_attributes():
    l = ['x', 'y', 'z']
    # for i in range(f_dc.shape[1]):
    #     l.append('f_dc_{}'.format(i))
    # l.append('opacity')
    # for i in range(scale.shape[1]):
    #     l.append('scale_{}'.format(i))
    # for i in range(rotation.shape[1]):
    #     l.append('rot_{}'.format(i))
    return l


def convert(src, dest):
    params = np.load(src)

    
    # for key in params.files:
    #     print(f"Array '{key}':")
    #     print(params[key])
    xyz = [vert[:7] for vert in params['data'] ]
    seg =  [vert[6] for vert in params['data'] if vert[6]==1 ]
    seg_b =  [vert[6] for vert in params['data'] if vert[6]==0 ]
    xyz_fg = []
    xyz_new=[]
    for ver in xyz:
        
        ver_fg = []
        for i, v in enumerate(ver.copy()):
            
            # print(i,v)
            # ver[3+i]=int(v)
            if i < 3:
                ver[i] = v

                if ver[6] == 1:
                    ver_fg.append(ver[i])
            elif i > 2 and i < 6:
                v=int(ver[6]*255)
                ver[i]=int(v)
                # v=int(v*255)
                # ver[i]=int(v)
        if len(ver_fg) == 3:
            xyz_fg.append(ver_fg)
        xyz_new.append(tuple(ver[:6]))
    center_fg = np.mean(np.asarray(xyz_fg), axis=0)
    xyz = np.asarray(xyz)
    center = np.mean(xyz[:,:3], axis=0)
    print("Foreground center:",center_fg)
    print("All scene center:", center)


    vertex=np.array(xyz_new, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
   
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(dest)


if __name__ == '__main__':
    # src ='/home/hamit/Softwares/Dynamic3DGaussians/data/juggle/init_pt_cld_org.npz'
    src = '/home/hamit/Softwares/Dynamic3DGaussians/data/2024-12-19_20-11-26_4096_180/init_pt_cld.npz'
    dest = '2024-12-19_20-11-26_4096.ply'
    convert(src, dest)
