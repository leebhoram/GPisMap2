""" some utility for call C++ code"""
# Adopted from
# https://github.com/geek-ai/MAgent/blob/master/python/magent/c_lib.py

import ctypes
import platform
from pathlib import Path


def _load_lib():
    """ Load libgpismap from <repo>/build/. """
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / 'build'

    system = platform.system()
    if system == 'Linux':
        path_to_so_file = build_dir / 'libgpismap.so'
    elif system == 'Darwin':
        path_to_so_file = build_dir / 'libgpismap.dylib'
    else:
        raise BaseException("unsupported system: " + system)

    lib = ctypes.CDLL(str(path_to_so_file), ctypes.RTLD_GLOBAL)
    return lib


def as_float_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def as_int32_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def as_bool_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))


_LIB = _load_lib()
