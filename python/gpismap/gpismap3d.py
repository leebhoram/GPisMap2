"""GPisMap 3D interface"""
from __future__ import absolute_import
import ctypes
import numpy as np
from .c_lib import _LIB , as_float_c_array


class GPisMap3D():
    """
    A class to interface GPisMap 3D C API.

    ...

    Attributes
    ----------
    No public attributes
    """

    def __init__(self):
        """
        Create new handle for GPisMap 2D object.

        Parameters
        ----------
            config_yaml : str [To-be-added]
                file name of config file

        Returns
        -------
        None
        """

        gm = ctypes.c_void_p()
        _LIB.create_gpm3d_instance(ctypes.byref(gm))
        self.gpmap = gm

    def set_cam_param(self, fx: float, fy: float, cx: float, cy: float, width:int, height: int):
        """
        Set camera parameters.

        Parameters
        ----------
            *using conventional symbols
            fx : focal length in x
            fy : focal length in y
            cx : camera center on image in x
            cy : camera center on image in y
            width : image width
            height : image height

        Returns
        -------
        None
        """

        _LIB.set_gpm3d_camparam(self.gpmap,
                ctypes.c_float(fx),
                ctypes.c_float(fy),
                ctypes.c_float(cx),
                ctypes.c_float(cy),
                width, height
            )

    def __del__(self):
        _LIB.delete_gpm3d_instance(self.gpmap)

    def reset(self):
        """
        Clear the map

        Parameters
        ----------
        NONE
        """
        _LIB.reset_gpm3d(self.gpmap)

    def update(self, depth:np.ndarray, position:np.ndarray , rotation:np.ndarray):
        """
        Update the map from depth camera-like observation

        Parameters
        ----------
            depth : np.ndarray
                depth values w.r.t the body frame

            position : np.ndarray (3 elements)
                position of the body in the map frame

            rotation : np.ndarray (flattened, 9 elements)
                orientation of the body in the map frame
                flattened rotation matrix

        Returns
        -------

        """

        assert depth.dtype == np.float32
        assert position.dtype == np.float32 and rotation.dtype == np.float32
        assert position.size == 3 and rotation.size == 9
        pose = np.concatenate((position,rotation),axis=None)
        res = _LIB.update_gpm3d(self.gpmap,
                as_float_c_array(depth.flatten('F')),
                depth.size,
                as_float_c_array(pose)
            )
        assert res == 1

    def test(self, test_x:np.ndarray):
        """
        Make inference using the map

        Parameters
        ----------
            test_x : np.ndarray
                x values to test (a 2D position on the map)

        Returns
        -------
            res : np.ndarray
                inference result
                # res[:,0] : sdf prediction
                # res[:,1] : grad x prediction
                # res[:,2] : grad y prediction
                # res[:,3] : grad z prediction
                # res[:,4] : sdf variance
                # res[:,5] : grad variance
        """

        assert test_x.dtype == np.float32
        dim = test_x.shape[1]
        leng = test_x.shape[0]

        res = np.empty((leng,2*(1+dim)),dtype=np.float32)

        _LIB.test_gpm3d(self.gpmap,
                as_float_c_array(test_x),
                dim,
                leng,
                as_float_c_array(res)
            )
        return res

    def get_sample_count(self):
        """
        Returns the number of samples of the current map

        Parameters
        ----------
        NONE

        Returns
        -------
            (int) number of map samples
        """
        return _LIB.get_sample_count_gpm3d(self.gpmap)

    def get_samples(self, grad=False, var=False):
        """
        Returns the samples of the current map

        Parameters
        ----------
            grad : bool [Optional]
                set True to get gradient of the samples
            var : bool [Optional]
                set True to get the variance of each sample

        Returns
        -------
            buf: np.ndarray  of shape [sample count x data_dim]
                if grad == True,
                    # buf[:,0:3] : sample's position
                    # buf[:,3:6] : sample's gradient (only if grad == True)
                    if var == True,
                        # buf[:,6] : variance of position
                        # buf[:,7] : variance of gradient
                else (i.e., grad == False),
                    # buf[:,0:3] : sample's position
                    if var == True,
                        # buf[:,3] : variance of position
        """

        count = self.get_sample_count()

        data_dim = 3
        if grad:
            data_dim = 6
        if var:
            data_dim = int(data_dim + data_dim/3)

        buf = np.empty((count, data_dim), dtype=np.float32)
        _LIB.get_samples_gpm3d(self.gpmap,
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          int(3),
                          int(count),
                          grad,
                          var)
        return buf
