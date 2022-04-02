
import shutil, itertools, time

from tempfile import TemporaryDirectory
from pytest import mark

from accelpy import Kernel, Accel, build_sharedlib, load_sharedlib, invoke_sharedlib
from testdata import get_testdata, assert_testdata

DEBUG = False

#test_vendors = ("cray", "ibm", "amd", "gnu")
test_vendors = ("cray",)
#test_vendors = ("amd",)

#test_codes = ("vecadd1d", "vecadd3d", "matmul")
test_codes = ("vecadd1d", "matmul", "vecadd3d")
#test_codes = ("matmul", "matmul", "vecadd1d")
#test_codes = ("matmul", )

test_langs = ("fortran",)

test_accels = ("omptarget", )
#test_accels = ("openacc", )
#test_accels = ("openmp", )
#test_accels = ("fortran", )

testcases = itertools.product(test_vendors, test_codes, test_langs, test_accels)

#test_vendors = ("gnu",)

def test_fortran():

    import os, sys
    import numpy as np

    lang = "fortran"

    N1 = 2
    N2 = 3

    X = np.ones(N1*N2).reshape((N1, N2), order="F")
    Y = np.ones(N1*N2).reshape((N1, N2), order="F") * 2
    Z = np.zeros(N1*N2).reshape((N1, N2), order="F")

    here = os.path.dirname(__file__)
    resdir = os.path.join(here, "res")

    libext = "dylib" if sys.platform == "darwin" else "so"

    srcdatafile = "ompdata.F90"
    outdatafile = "libompdata." + libext
    srckernelfile = "ompkernel.F90" 
    outkernelfile = "libompkernel." + libext

    srcdatapath = os.path.join(resdir, srcdatafile)
    outdatapath = os.path.join(resdir, outdatafile)

    srckernelpath = os.path.join(resdir, srckernelfile)
    outkernelpath = os.path.join(resdir, outkernelfile)

    # build acceldata
    with TemporaryDirectory() as workdir:

        if os.path.exists(outdatapath):
            os.remove(outdatapath)
        
        shutil.copy(srcdatapath, workdir)
        datapath = build_sharedlib(srcdatafile, outdatafile, workdir,
                        vendor=test_vendors)

        assert os.path.isfile(datapath)

        # load acceldata
        libdata = load_sharedlib(datapath)
        assert libdata is not None

        # invoke function in acceldata
        resdata = invoke_sharedlib(lang, libdata, "dataenter", X, Y, Z)
        assert resdata == 0

        # build kernel
        if os.path.exists(outkernelpath):
            os.remove(outkernelpath)

        shutil.copy(srckernelpath, workdir)
        kernelpath = build_sharedlib(srckernelfile, outkernelfile, workdir,
                        vendor=test_vendors)

        assert os.path.isfile(kernelpath)

        # load kernel
        libkernel = load_sharedlib(kernelpath)
        assert libkernel is not None

        # invoke function in kernel
        reskernel = invoke_sharedlib(lang, libkernel, "runkernel", X, Y, Z)
        assert reskernel == 0

        # invoke function in acceldata
        resdata = invoke_sharedlib(lang, libdata, "dataexit", Z)
        assert resdata == 0

        # check result
        assert np.array_equal(Z, X+Y)

@mark.parametrize("vendor, code, lang, accel", testcases)
def test_omptarget2(vendor, code, lang, accel):

    data, knl  = get_testdata(code, lang)

    # TODO: generate data var names
    acc = Accel(**data, vendor=vendor, accel=accel, _debug=DEBUG)

    # TODO: testif data var names are used in launch
    args = []
    args.extend(data["copyinout"])
    args.extend(data["copyin"])
    args.extend(data["copyout"])
    args.extend(data["alloc"])

    acc.launch(Kernel(knl), *args)

    acc.stop()


    assert_testdata(code, data)



