import cv2
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes

_parula_data = [[0.2081, 0.1663, 0.5292], 
                [0.2116238095, 0.1897809524, 0.5776761905], 
                [0.212252381, 0.2137714286, 0.6269714286], 
                [0.2081, 0.2386, 0.6770857143], 
                [0.1959047619, 0.2644571429, 0.7279], 
                [0.1707285714, 0.2919380952, 0.779247619], 
                [0.1252714286, 0.3242428571, 0.8302714286], 
                [0.0591333333, 0.3598333333, 0.8683333333], 
                [0.0116952381, 0.3875095238, 0.8819571429], 
                [0.0059571429, 0.4086142857, 0.8828428571], 
                [0.0165142857, 0.4266, 0.8786333333], 
                [0.032852381, 0.4430428571, 0.8719571429], 
                [0.0498142857, 0.4585714286, 0.8640571429], 
                [0.0629333333, 0.4736904762, 0.8554380952], 
                [0.0722666667, 0.4886666667, 0.8467], 
                [0.0779428571, 0.5039857143, 0.8383714286], 
                [0.079347619, 0.5200238095, 0.8311809524], 
                [0.0749428571, 0.5375428571, 0.8262714286], 
                [0.0640571429, 0.5569857143, 0.8239571429], 
                [0.0487714286, 0.5772238095, 0.8228285714], 
                [0.0343428571, 0.5965809524, 0.819852381], 
                [0.0265, 0.6137, 0.8135], 
                [0.0238904762, 0.6286619048, 0.8037619048], 
                [0.0230904762, 0.6417857143, 0.7912666667], 
                [0.0227714286, 0.6534857143, 0.7767571429], 
                [0.0266619048, 0.6641952381, 0.7607190476], 
                [0.0383714286, 0.6742714286, 0.743552381], 
                [0.0589714286, 0.6837571429, 0.7253857143], 
                [0.0843, 0.6928333333, 0.7061666667], 
                [0.1132952381, 0.7015, 0.6858571429], 
                [0.1452714286, 0.7097571429, 0.6646285714], 
                [0.1801333333, 0.7176571429, 0.6424333333], 
                [0.2178285714, 0.7250428571, 0.6192619048], 
                [0.2586428571, 0.7317142857, 0.5954285714], 
                [0.3021714286, 0.7376047619, 0.5711857143], 
                [0.3481666667, 0.7424333333, 0.5472666667], 
                [0.3952571429, 0.7459, 0.5244428571], 
                [0.4420095238, 0.7480809524, 0.5033142857], 
                [0.4871238095, 0.7490619048, 0.4839761905], 
                [0.5300285714, 0.7491142857, 0.4661142857], 
                [0.5708571429, 0.7485190476, 0.4493904762],
                [0.609852381, 0.7473142857, 0.4336857143], 
                [0.6473, 0.7456, 0.4188], 
                [0.6834190476, 0.7434761905, 0.4044333333], 
                [0.7184095238, 0.7411333333, 0.3904761905], 
                [0.7524857143, 0.7384, 0.3768142857], 
                [0.7858428571, 0.7355666667, 0.3632714286], 
                [0.8185047619, 0.7327333333, 0.3497904762], 
                [0.8506571429, 0.7299, 0.3360285714], 
                [0.8824333333, 0.7274333333, 0.3217], 
                [0.9139333333, 0.7257857143, 0.3062761905], 
                [0.9449571429, 0.7261142857, 0.2886428571], 
                [0.9738952381, 0.7313952381, 0.266647619], 
                [0.9937714286, 0.7454571429, 0.240347619], 
                [0.9990428571, 0.7653142857, 0.2164142857], 
                [0.9955333333, 0.7860571429, 0.196652381], 
                [0.988, 0.8066, 0.1793666667], 
                [0.9788571429, 0.8271428571, 0.1633142857], 
                [0.9697, 0.8481380952, 0.147452381], 
                [0.9625857143, 0.8705142857, 0.1309], 
                [0.9588714286, 0.8949, 0.1132428571], 
                [0.9598238095, 0.9218333333, 0.0948380952], 
                [0.9661, 0.9514428571, 0.0755333333], 
                [0.9763, 0.9831, 0.0538]]

from matplotlib.colors import ListedColormap
parula = ListedColormap(_parula_data, name='parula')

def alpha_blend(value: np.ndarray, alpha: np.ndarray, normalize_between=[-0.5, 0.5]):
    assert value.shape[0] == alpha.shape[0]
    assert value.shape[1] == alpha.shape[1]
    alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))   
    if normalize_between is not None:
        min_v = np.min(normalize_between)
        max_v = np.max(normalize_between)
        value = (value - min_v) / (max_v - min_v)

    rgb = parula(value)
    rgb = np.multiply(alpha, rgb)
    background = np.ones_like(rgb)
    background = np.multiply(1.0-alpha, background)
    return cv2.rotate(cv2.add(rgb, background), cv2.ROTATE_90_COUNTERCLOCKWISE)


def set_axes3d_equal(ax):
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


def show_map_2d(gp, meshgrid_xy):    
    xg, yg = meshgrid_xy
    xtest = np.stack((xg.flatten(), yg.flatten()), axis=-1).astype(np.float32)

    f = gp.test(xtest)
    # f[:,0] : sdf prediction
    # f[:,1] : grad x prediction
    # f[:,2] : grad y prediction
    # f[:,3] : sdf variance
    # f[:,4] : grad x variance
    # f[:,5] : grad y variance
    sdf = np.reshape(f[:,0]+0.2,xg.shape)
    va = np.max(f[:,3]) - np.reshape(f[:,3],xg.shape);
    Im = alpha_blend(sdf,va)
    fig, ax = plt.subplots(1, 1)
    plt.imshow(Im)
    samples = gp.get_samples()
    plt.scatter((samples[:,0]-xg[0,0])/(xg[-1,-1]-xg[0,0])*xg.shape[0], 
                (1-(samples[:,1]-yg[0,0])/(yg[-1,-1]-yg[0,0]))*yg.shape[1], s=1, c='k')
    plt.tight_layout()
    plt.tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)
    plt.show()


def show_mesh_3d(gp3d, meshgrid_xyz, var_thre=0.1):    
    xg, yg, zg = meshgrid_xyz
    xtest = np.stack((xg.flatten(), yg.flatten(), zg.flatten()), axis=-1).astype(np.float32)

    f = gp3d.test(xtest)
    # f[:,0] : sdf prediction
    # f[:,1] : grad x prediction
    # f[:,2] : grad y prediction
    # f[:,3] : grad z prediction
    # f[:,4] : sdf variance
    # f[:,5] : grad x variance
    # f[:,6] : grad y variance
    # f[:,7] : grad z variance  
    high_confidence_index = np.reshape(np.array(f[:,4]<var_thre),xg.shape)
    sdf=np.reshape(f[:,0]+0.2,xg.shape)
    var = np.reshape(f[:,4],xg.shape)
    verts, faces, _, _ = marching_cubes(sdf, 0.0, 
            mask=high_confidence_index)

    # rescale vertex coordinate 
    # still looks a bit off
    mins = np.array([xg[0,0,0], yg[0,0,0], zg[0,0,0]])
    maxes = np.array([xg[-1,-1,-1], yg[-1,-1,-1], zg[-1,-1,-1]])
    ranges = maxes - mins
    verts_scaled = (verts + 0.5) * ranges / np.array(xg.shape) + mins
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    patches = ax.plot_trisurf(verts_scaled[:, 0], verts_scaled[:,1], faces, verts_scaled[:, 2], 
                    edgecolor=[0.1,0.1,0.1],
                    lw=0.1)
    patches.set_facecolor([0.5,0.5,0.5])
    samples = gp3d.get_samples()
    ax.scatter(samples[:,0],samples[:,1],samples[:,2], s=1, c='k')
    ax.set_box_aspect([1,1,1])
    set_axes3d_equal(ax)
    plt.show()
