import argparse
import os
import numpy as np
import yaml
import time
from PIL import Image
from tqdm import tqdm
from gpismap import GPisMap3D
from  util.visualization import MeshStream3D, MeshStream3DPyVista

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyvista", action="store_true",
                        help="Render the final mesh with PyVista (translucent)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Opacity for the PyVista renderer (0..1)")
    args = parser.parse_args()


    depthpath = '../data/3D/bigbird_detergent/masked_depth'
    poses = np.loadtxt('../data/3D/bigbird_detergent/pose/poses.txt')
    with open('../data/3D/bigbird_detergent/camera_param.yaml') as f:
        cam_params = yaml.load(f, Loader=yaml.FullLoader)

    frameid = poses[:,0].astype(np.int32)
    camid = poses[:,1].astype(np.int32)
    poses = np.array(poses[:,2:], dtype=np.float32)

    # for visualization
    xmin = -0.07
    xmax = 0.13
    ymin = -0.1
    ymax = 0.14
    zmin = 0
    zmax = 0.28
    test_intv = 0.01

    x = np.linspace(xmin, xmax, int((xmax-xmin)/test_intv))
    y = np.linspace(ymin, ymax, int((ymax-ymin)/test_intv))
    z = np.linspace(zmin, zmax, int((zmax-zmin)/test_intv))
    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')

    gp = GPisMap3D()

    if args.pyvista:
        stream = MeshStream3DPyVista((xg, yg, zg), alpha=args.alpha)
    else:
        stream = MeshStream3D((xg, yg, zg))

    pbar = tqdm(range(len(frameid)), desc="frames", unit="frame")
    for k in pbar:
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
        pbar.set_postfix(update_s=f"{toc - tic:0.3f}")

        stream.update(gp)


    input("Press Enter to end...")
    stream.close()
    gp.reset()

if __name__ == "__main__":
    main()
