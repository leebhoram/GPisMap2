"""GPisMap 2D interface"""
from __future__ import absolute_import
import ctypes
import os
import importlib
from .c_lib import _LIB , as_float_c_array # , as_int32_c_array
import numpy as np

class GPisMap2D():
    def __init__(self, config_yaml:str=None, **kwargs):
        # create new instance
        gm = ctypes.c_void_p()
        _LIB.create_gpm_instance(ctypes.byref(gm))
        self.gpmap = gm

        config_dict = {}
        if config_yaml is not None:
            import yaml
            print(f"Loading {config_yaml}")
            config_dict = yaml.load(config_yaml)
        for key, val in config_dict.items():
            _LIB.config_gpm(self.gpmap, key.encode("ascii"), ctypes.byref(ctypes.c_float(val)))

    def __del__(self):
        _LIB.delete_gpm_instance(self.gpmap)

    def reset(self):
        _LIB.reset_gpm(self.gpmap)

    # int update_gpm(GPMHandle gh, float * datax,  float * dataf, int N, float* pose); // pose[6]
    def update(self, datax:np.ndarray, dataf:np.ndarray, t , R):
        assert datax.dtype == np.float32
        assert dataf.dtype == np.float32
        assert t.dtype == np.float32 and R.dtype == np.float32
        assert len(datax) == len(dataf) 
        N = len(datax)
        assert len(t) == 2 and len(R) == 4
        pose = np.concatenate((t,R),axis=None,dtype=np.float32)
        res = _LIB.update_gpm(self.gpmap, 
                as_float_c_array(datax),
                as_float_c_array(dataf),
                N,
                as_float_c_array(pose)
            )

    def test(self, x:np.ndarray):
        assert x.dtype == np.float32
        dim = x.shape[1]
        leng = x.shape[0]

        res = np.empty((leng,2*(1+dim)),dtype=np.float32)
        
        _LIB.test_gpm(self.gpmap, 
                as_float_c_array(x),
                dim,
                leng,
                as_float_c_array(res)
            )
        return res

    def get_sample_count(self):
        return _LIB.get_sample_count_gpm(self.gpmap)

    def get_samples(self, grad=False, var=False):
        count = self.get_sample_count()

        data_dim = 2
        if grad:
            data_dim = 4
        if var:
            data_dim = int(data_dim + data_dim/2)

        buf = np.empty((count, data_dim), dtype=np.float32)      
        _LIB.get_samples_gpm(self.gpmap,
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          int(2),
                          int(count),
                          grad,
                          var);
        return buf

                
    
