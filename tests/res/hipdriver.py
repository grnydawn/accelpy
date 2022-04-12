
import os
import numpy as np
from accelpy import build_sharedlib, load_sharedlib, invoke_sharedlib

N1 = 2
N2 = 3

X = np.ones(N1*N2, order="C").reshape((N1, N2))
Y = np.ones(N1*N2, order="C").reshape((N1, N2)) * 2
Z = np.zeros(N1*N2, order="C").reshape((N1, N2))

here = os.path.dirname(__file__)

vendor = "amd"
lang = "cpp"
accel = "hip"

srcdatafile = "hipdata.cpp"
outdatafile = "libhipdata.so"
srckernelfile = "hipkernel.cpp"
outkernelfile = "libhipkernel.so"

srcdatapath = os.path.join(here, srcdatafile)
outdatapath = os.path.join(here, outdatafile)
srckernelpath = os.path.join(here, srckernelfile)
outkernelpath = os.path.join(here, outkernelfile)

# build acceldata
output = "-o " + outdatapath
build_sharedlib(srcdatafile, outdatafile, here, vendor=vendor, lang=lang, accel=accel)
assert os.path.isfile(outdatapath)

# load acceldata
libdata = load_sharedlib(outdatapath)
assert libdata is not None

# invoke function in acceldata
resdata = invoke_sharedlib(lang, libdata, "dataenter", X, Y, Z)
assert resdata == 0

# build kernel
build_sharedlib(srckernelfile, outkernelfile, here, vendor=vendor, lang=lang, accel=accel)
assert os.path.isfile(outkernelpath)

# load kernel
libkernel = load_sharedlib(outkernelpath)
assert libkernel is not None

# invoke function in kernel
reskernel = invoke_sharedlib(lang, libkernel, "runkernel", X, Y, Z)
assert reskernel == 0

# invoke function in acceldata
resdata = invoke_sharedlib(lang, libdata, "dataexit", Z)
assert resdata == 0

# check result
assert np.array_equal(Z, X+Y)

print("SUCCESS")
