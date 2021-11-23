import os, sys
import numpy as np
from numpy.ctypeslib import load_library, ndpointer
from ctypes import c_int, c_double, byref, POINTER

def debug():
    import pdb; pdb.set_trace()

lib = load_library(sys.argv[1], ".")

N = 10

X = np.ones(N)
Y = np.ones(N) * 2
Z = np.zeros(N)

attr = np.array((N, 1))

add1d = getattr(lib, "add1d")
add1d.restype = c_int
add1d.argtypes = [POINTER(c_int),
        ndpointer(attr.dtype, flags='aligned, contiguous'),
        ndpointer(X.dtype, flags='aligned, contiguous'),
        ndpointer(Y.dtype, flags='aligned, contiguous'),
        ndpointer(Z.dtype, flags='aligned, contiguous')]

add1d(byref(c_int(attr.size)), attr, X, Y, Z)

assert np.array_equal(Z, X + Y)
debug()
