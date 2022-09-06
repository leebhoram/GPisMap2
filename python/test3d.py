import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import time
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from gpismap import GPisMap3D

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main():   

    depthpath = '../data/3D/bigbird_detergent/masked_depth'
    poses = np.loadtxt('../data/3D/bigbird_detergent/pose/poses.txt')
    with open('../data/3D/bigbird_detergent/camera_param.yaml') as f:
        cam_params = yaml.load(f, Loader=yaml.FullLoader)
    
    frameid = poses[:,0].astype(np.int32)
    camid = poses[:,1].astype(np.int32)
    poses = np.array(poses[:,2:], dtype=np.float32)

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
    xtest = np.stack((xg.flatten(), yg.flatten(), zg.flatten()), axis=-1).astype(np.float32)

    gp = GPisMap3D()

    for k in range(len(frameid)):
        print(f"#frame: {k}")
        I = Image.open(os.path.join(depthpath, f'frame{frameid[k]}_cam{camid[k]}.png'))
        I = 0.0001*np.asarray(I,dtype=np.float32) # 10 mm to meter
        T = np.reshape(poses[k],(4,4))
        tr = T[0:3,3].flatten()
        Rot = T[0:3,0:3].flatten()

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

        if k == 0: #len(frameid)-1:
            # visualize
            print("Visualizing...")
            f = gp.test(xtest)
            sdf=np.reshape(f[:,0]+0.2,xg.shape)
            #var = np.reshape(f[:,5],xg.shape)
            iso_val=0.0
            verts, faces, _, _ = marching_cubes(sdf, iso_val, spacing=(test_intv,test_intv,test_intv))
            f = gp.test(verts.astype(np.float32))
            var = f[:,4].flatten()
            var = (var-np.min(var))/(np.max(var)-np.min(var)+0.00001)
            var = np.clip(var,0,1)
            color = 0.5*np.ones((var.size,4),dtype=np.float32)
            color[:,3] = var
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            patches = ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],# facecolor=color,
                            lw=0.5)
            patches.set_facecolor(color)
            ax.set_box_aspect([1,1,1])
            set_axes_equal(ax)
            plt.show()
            break

    gp.reset()
    input("Press Enter to end...")

if __name__ == "__main__":
    main()
