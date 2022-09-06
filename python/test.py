from scipy.io import loadmat
import numpy as np
from gpismap import GPisMap2D
import matplotlib.pyplot as plt

def main():
    data = loadmat('../data/2D/gazebo1.mat')

    poses = np.array(data['poses'],dtype=np.float32)
    thetas = np.array(data['thetas'],dtype=np.float32).flatten()
    ranges = np.array(data['ranges'],dtype=np.float32)

    xmin = -5;
    xmax = 20;
    ymin = -15;
    ymax = 5;
    test_intv = 0.1;

    x = np.linspace(xmin, xmax, int((xmax-xmin)/test_intv))
    y = np.linspace(ymin, ymax, int((ymax-ymin)/test_intv))
    xg, yg = np.meshgrid(x, y, indexing='ij')
    xtest = np.stack((xg.flatten(), yg.flatten()), axis=-1).astype(np.float32)
    
    skip = 100
    initframe = 100 # first 100 frames are almost static...
    lastframe = 1000 #int((np.floor((poses.shape[0]-initframe)/skip))*skip+initframe)

    gp = GPisMap2D()

    for nframe in range(initframe,lastframe, skip):
        print(f"#frame: {nframe}")
        tr = poses[nframe,0:2]
        phi = poses[nframe,2]
        Rot = np.array([np.cos(phi), np.sin(phi), -np.sin(phi), np.cos(phi)])     
        range_obs = ranges[nframe,:].flatten()

        gp.update(thetas, range_obs, tr , Rot)
                       
        if nframe % 10 == 0:
            f = gp.test(xtest)
            sdf=np.reshape(f[:,0]+0.2,xg.shape)
            fig, ax = plt.subplots(1, 1)
            ax.pcolor(xg, yg, sdf)
            plt.show()
        break
    

if __name__ == "__main__":
    main()
