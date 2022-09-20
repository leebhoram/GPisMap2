"""GPisMap 2D interface"""
from __future__ import absolute_import
import ctypes
import yaml
import numpy as np
from .c_lib import _LIB , as_float_c_array # , as_int32_c_array


class GPisMap2D():
    """
    A class to interface GPisMap 2D C API.

    ...

    Attributes
    ----------
    No public attributes
    """

    def __init__(self, config_yaml:str=""):
        """
        Create new handle for GPisMap 2D object.

        Parameters
        ----------
            config_yaml : str
                file name of config file

        Returns
        -------
        None
        """

        gp_map = ctypes.c_void_p()
        _LIB.create_gpm_instance(ctypes.byref(gp_map))
        self.gp_map = gp_map

        config_dict = {}
        if len(config_yaml) > 0:
            print(f"Loading {config_yaml}")
            config_dict = yaml.load(config_yaml)
        for key, val in config_dict.items():
            _LIB.config_gpm(self.gp_map, key.encode("ascii"), ctypes.byref(ctypes.c_float(val)))

    def __del__(self):
        _LIB.delete_gpm_instance(self.gp_map)

    def reset(self):
        """
        Clear the map

        Parameters
        ----------
        NONE
        """
        _LIB.reset_gpm(self.gp_map)

    def update(self, datax:np.ndarray,
               dataf:np.ndarray,
               position: np.ndarray,
               rotation: np.ndarray):
        """
        Update the map from lidar-like observation data

        Parameters
        ----------
            datax : np.ndarray
                x values of input (angular position) w.r.t the body frame

            dataf : np.ndarray
                f values of input (distance) w.r.t the body frame

            position : np.ndarray
                position of the body in the map frame

            rotation : np.ndarray (flattened)
                orientation of the body in the map frame
                flattened rotation matrix

        Returns
        -------

        """

        assert datax.dtype == np.float32
        assert dataf.dtype == np.float32
        assert position.dtype == np.float32 and rotation.dtype == np.float32
        assert len(datax) == len(dataf)
        assert len(position) == 2 and len(rotation) == 4
        pose = np.concatenate((position,rotation), axis=None)
        res = _LIB.update_gpm(self.gp_map,
                as_float_c_array(datax),
                as_float_c_array(dataf),
                len(datax),
                as_float_c_array(pose)
            )
        assert res == 1

    def test(self, test_x: np.ndarray):
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
                # res[:,3] : sdf variance
                # res[:,4] : grad variance
        """

        assert test_x.dtype == np.float32
        dim = test_x.shape[1]
        leng = test_x.shape[0]

        res = np.empty((leng,2*(1+dim)),dtype=np.float32)

        _LIB.test_gpm(self.gp_map,
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

        return _LIB.get_sample_count_gpm(self.gp_map)

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
                    # buf[:,0:2] : sample's position
                    # buf[:,2:4] : sample's gradient (only if grad == True)
                    if var == True,
                        # buf[:,5] : variance of position
                        # buf[:,6] : variance of gradient
                else (i.e., grad == False),
                    # buf[:,0:2] : sample's position
                    if var == True,
                        # buf[:,2] : variance of position
        """

        count = self.get_sample_count()

        data_dim = 2
        if grad:
            data_dim = 4
        if var:
            data_dim = int(data_dim + data_dim/2)

        buf = np.empty((count, data_dim), dtype=np.float32)
        _LIB.get_samples_gpm(self.gp_map,
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          int(2),
                          int(count),
                          grad,
                          var)
        return buf
