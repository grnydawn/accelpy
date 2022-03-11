"""Accelpy accel module"""

import os

from accelpy.util import load_sharedlib, invoke_sharedlib
from accelpy.compile import build_sharedlib

class Accel:

    def __init__(self, srcdatapath, outdatapath, srckernelpath, outkernelpath,
                    vendor=None):

        self.srcdatapath =srcdatapath
        self.outdatapath = outdatapath
        self.srckernelpath = srckernelpath
        self.outkernelpath = outkernelpath
        self.vendor = vendor


    def run(self, *vargs):

        # build acceldata
        build_sharedlib(self.srcdatapath, self.outdatapath, vendor=self.vendor)
        assert os.path.isfile(self.outdatapath)

        # load acceldata
        libdata = load_sharedlib(self.outdatapath)
        assert libdata is not None

        # invoke function in acceldata
        resdata = invoke_sharedlib(libdata, "dataenter", *vargs)
        assert resdata == 0

        # build kernel
        build_sharedlib(self.srckernelpath, self.outkernelpath, vendor=self.vendor)
        assert os.path.isfile(self.outkernelpath)

        # load kernel
        libkernel = load_sharedlib(self.outkernelpath)
        assert libkernel is not None

        # invoke function in kernel
        reskernel = invoke_sharedlib(libkernel, "runkernel", *vargs)
        assert reskernel == 0

        # invoke function in acceldata
        resdata = invoke_sharedlib(libdata, "dataexit", vargs[-1])
        assert resdata == 0

