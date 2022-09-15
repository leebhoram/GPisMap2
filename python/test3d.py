import os
import numpy as np
import yaml
import time
from PIL import Image
from gpismap import GPisMap3D
from  util.visualization import show_mesh_3d

def main():   

    depthpath = '../data/3D/bigbird_detergent/masked_depth'
    poses = np.loadtxt('../data/3D/bigbird_detergent/pose/poses.txt')
    with open('../data/3D/bigbird_detergent/camera_param.yaml') as f:
        cam_params = yaml.load(f, Loader=yaml.FullLoader)
    
    frameid = poses[:,0].astype(np.int32)
    camid = poses[:,1].astype(np.int32)
    poses = np.array(poses[:,2:], dtype=np.float32)

    # for visualization
    xmin = -0.07;
    xmax = 0.13;
    ymin = -0.1;
    ymax = 0.14;
    zmin = 0;
    zmax = 0.28;
    test_intv = 0.01;

    x = np.linspace(xmin, xmax, int((xmax-xmin)/test_intv))
    y = np.linspace(ymin, ymax, int((ymax-ymin)/test_intv))
    z = np.linspace(zmin, zmax, int((zmax-zmin)/test_intv))
    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')

    gp = GPisMap3D()

    for k in range(len(frameid)):
        print(f"#frame: {k}")
        I = Image.open(os.path.join(depthpath, f'frame{frameid[k]}_cam{camid[k]}.png'))
        I = 0.0001*np.asarray(I,dtype=np.float32) # 10 mm to meter
        T = np.reshape(poses[k],(4,4))
        tr = T[0:3,3].flatten()
        Rot = T[0:3,0:3].flatten()

        # can be called only once if the camera param is fixed
        # in this demo, multiple cameras are used.
        cam_param = cam_params[camid[k]-1]
        gp.set_cam_param(cam_param['fx'], cam_param['fy'],
                         cam_param['cx'], cam_param['cy'], 
                         int(cam_param['width']),
                         int(cam_param['height'])
                        )
        tic = time.perf_counter()
        gp.update(I, tr, Rot)
        toc = time.perf_counter()
        print(f"Elapsed time: {toc - tic:0.4f} seconds...")
        
        show_mesh_3d(gp, (xg, yg, zg))
        input("Press Enter to continue...")

    gp.reset()
    input("Press Enter to end...")

if __name__ == "__main__":
    main()
