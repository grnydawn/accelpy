import os, sys
import numpy as np
from numpy.ctypeslib import load_library
from ctypes import c_int, c_double, byref, POINTER

def debug():
    import pdb; pdb.set_trace()

lib = load_library(sys.argv[1], ".")

x = 1.0
y = 2.0
z = 3.0

add1d = getattr(lib, "add1d")
add1d.restype = c_int
#add1d.argtypes = [byref(c_double()), byref(c_double()), byref(c_double())]
add1d.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double)]

add1d(byref(c_double(x)), byref(c_double(y)), byref(c_double(z)))

assert z == x + y

debug()


