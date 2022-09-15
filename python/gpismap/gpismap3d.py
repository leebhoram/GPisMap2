"""GPisMap 3D interface"""
from __future__ import absolute_import
import ctypes
import os
import importlib
from .c_lib import _LIB , as_float_c_array 
import numpy as np

class GPisMap3D():
    def __init__(self, config_yaml:str=None, **kwargs):
        # create new instance
        gm = ctypes.c_void_p()
        _LIB.create_gpm3d_instance(ctypes.byref(gm))
        self.gpmap = gm

    def set_cam_param(self, fx: float, fy: float, cx: float, cy: float, width:int, height: int):
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
        _LIB.reset_gpm3d(self.gpmap)
   
    def update(self, depth:np.ndarray, t:np.ndarray , R:np.ndarray):
        assert depth.dtype == np.float32
        assert t.dtype == np.float32 and R.dtype == np.float32
        N = depth.size
        assert t.size == 3 and R.size == 9
        pose = np.concatenate((t,R),axis=None,dtype=np.float32)
        res = _LIB.update_gpm3d(self.gpmap, 
                as_float_c_array(depth.flatten('F')),
                N,
                as_float_c_array(pose)
            )

    def test(self, x:np.ndarray):
        assert x.dtype == np.float32
        dim = x.shape[1]
        leng = x.shape[0]

        res = np.empty((leng,2*(1+dim)),dtype=np.float32)
        
        _LIB.test_gpm3d(self.gpmap, 
                as_float_c_array(x),
                dim,
                leng,
                as_float_c_array(res)
            )
        return res
    
    def get_sample_count(self):
        return _LIB.get_sample_count_gpm3d(self.gpmap)

    def get_samples(self, grad=False, var=False):
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
                          var);
        return buf
