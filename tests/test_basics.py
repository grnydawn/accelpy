
import os, sys
import numpy as np
from accelpy import Accel, build_sharedlib, load_sharedlib, invoke_sharedlib

N1 = 2
N2 = 3

X = np.ones(N1*N2, order="F").reshape((N1, N2))
Y = np.ones(N1*N2, order="F").reshape((N1, N2)) * 2
Z = np.zeros(N1*N2, order="F").reshape((N1, N2))

here = os.path.dirname(__file__)
resdir = os.path.join(here, "res")

libext = "dylib" if sys.platform == "darwin" else "so"

srcdatapath = os.path.join(resdir, "ompdata.F90")
outdatapath = os.path.join(resdir, "libompdata." + libext)
srckernelpath = os.path.join(resdir, "ompkernel.F90")
outkernelpath = os.path.join(resdir, "libompkernel." + libext)

def test_omptarget1():

    # build acceldata
    build_sharedlib(srcdatapath, outdatapath, vendor="cray")
    assert os.path.isfile(outdatapath)

    # load acceldata
    libdata = load_sharedlib(outdatapath)
    assert libdata is not None

    # invoke function in acceldata
    resdata = invoke_sharedlib(libdata, "dataenter", X, Y, Z)
    assert resdata == 0

    # build kernel
    build_sharedlib(srckernelpath, outkernelpath, vendor="cray")
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

def test_omptarget2():

    accel = Accel(srcdatapath, outdatapath, srckernelpath, outkernelpath, vendor="cray")
    accel.run(X, Y, Z)

#    # build acceldata
#    build_sharedlib(srcdatapath, outdatapath, vendor="cray")
#    assert os.path.isfile(outdatapath)
#
#    # load acceldata
#    libdata = load_sharedlib(outdatapath)
#    assert libdata is not None
#
#    # invoke function in acceldata
#    resdata = invoke_sharedlib(libdata, "dataenter", X, Y, Z)
#    assert resdata == 0
#
#    # build kernel
#    build_sharedlib(srckernelpath, outkernelpath, vendor="cray")
#    assert os.path.isfile(outkernelpath)
#
#    # load kernel
#    libkernel = load_sharedlib(outkernelpath)
#    assert libkernel is not None
#
#    # invoke function in kernel
#    reskernel = invoke_sharedlib(libkernel, "runkernel", X, Y, Z)
#    assert reskernel == 0
#
#    # invoke function in acceldata
#    resdata = invoke_sharedlib(libdata, "dataexit", Z)
#    assert resdata == 0

    # check result
    assert np.array_equal(Z, X+Y)
