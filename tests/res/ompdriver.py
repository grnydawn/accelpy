
import os
import numpy as np
from accelpy import build_sharedlib, load_sharedlib, invoke_sharedlib

N1 = 2
N2 = 3

X = np.ones(N1*N2, order="F").reshape((N1, N2))
Y = np.ones(N1*N2, order="F").reshape((N1, N2)) * 2
Z = np.zeros(N1*N2, order="F").reshape((N1, N2))

here = os.path.dirname(__file__)

srcdatapath = os.path.join(here, "ompdata.F90")
outdatapath = os.path.join(here, "libompdata.so")
srckernelpath = os.path.join(here, "ompkernel.F90")
outkernelpath = os.path.join(here, "libompkernel.so")

compile_data = "ftn -shared -fPIC -fopenmp {output} {moddir} {inputs}"
compile_kernel = "ftn -shared -fPIC -fopenmp {output} {moddir} {inputs}"

# build acceldata
output = "-o " + outdatapath
moddir = "-J " + here
build_sharedlib(compile_data.format(inputs=srcdatapath, output=output,
                    moddir=moddir))
assert os.path.isfile(outdatapath)

# load acceldata
libdata = load_sharedlib(outdatapath)
assert libdata is not None

# invoke function in acceldata
resdata = invoke_sharedlib(libdata, "dataenter", X, Y, Z)
assert resdata == 0

# build kernel
output = "-o " + outkernelpath
moddir = "-J " + here
build_sharedlib(compile_kernel.format(inputs=srckernelpath, output=output,
                    moddir=moddir))
assert os.path.isfile(outkernelpath)

# load kernel
libkernel = load_sharedlib(outkernelpath)
assert libkernel is not None

# invoke function in kernel
reskernel = invoke_sharedlib(libkernel, "runkernel", X, Y, Z)
assert reskernel == 0

# invoke function in acceldata
resdata = invoke_sharedlib(libdata, "dataexit", Z)
assert resdata == 0

# check result
assert np.array_equal(Z, X+Y)

print("SUCCESS")
