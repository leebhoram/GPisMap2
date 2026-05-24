import time
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from gpismap import GPisMap2D
from  util.visualization import MapStream2D


def main():
    data = loadmat('../data/2D/gazebo1.mat')

    poses = np.array(data['poses'],dtype=np.float32)
    thetas = np.array(data['thetas'],dtype=np.float32).flatten()
    ranges = np.array(data['ranges'],dtype=np.float32)

    # for visualization
    xmin = -5
    xmax = 20
    ymin = -15
    ymax = 5
    test_intv = 0.1
    x = np.linspace(xmin, xmax, int((xmax-xmin)/test_intv))
    y = np.linspace(ymin, ymax, int((ymax-ymin)/test_intv))
    xg, yg = np.meshgrid(x, y, indexing='ij')


    skip = 100
    initframe = 100 # first 100 frames are almost static...
    lastframe = poses.shape[0]

    gp = GPisMap2D()
    stream = MapStream2D((xg, yg))

    pbar = tqdm(range(initframe, lastframe, skip), desc="frames", unit="frame")
    for nframe in pbar:
        tr = poses[nframe,0:2]
        phi = poses[nframe,2]
        Rot = np.array([np.cos(phi), np.sin(phi), -np.sin(phi), np.cos(phi)])
        range_obs = ranges[nframe,:].flatten()

        tic = time.perf_counter()
        gp.update(thetas, range_obs, tr , Rot)
        toc = time.perf_counter()
        pbar.set_postfix(frame=nframe, update_s=f"{toc - tic:0.3f}")

        stream.update(gp, tr, phi)

    input("Press Enter to end...")
    stream.close()
    gp.reset()


if __name__ == "__main__":
    main()
