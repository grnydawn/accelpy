"""accelpy Accel module"""

import abc, os, tempfile

from ctypes import CDLL, RTLD_GLOBAL
from numpy.ctypeslib import ndpointer
from collections import OrderedDict
from accelpy.const import (version, NOCACHE, MEMCACHE, FILECACHE, NODEBUG,
                            MINDEBUG, MAXDEBUG, NOPROF, MINPROF, MAXPROF,
                            accel_priority)
from accelpy.util import Object, gethash, get_config, set_config, pack_arguments
from accelpy.compiler import get_compilers

from accelpy.cache import slib_cache

class AccelDataBase(Object):

    avails = OrderedDict()
    offload = False

    # get map info, build & load slib, run mapping, hand over slib to kernels
    def __init__(self, *kernels, mapto=[], maptofrom=[], mapfrom=[],
                    mapalloc=[], cache=MEMCACHE, profile=NOPROF, debug=NODEBUG):

        self.kernels = kernels
        self.debug = debug
        self.profile = profile
        self.cache = cache

        if not self.offload:
            for kernel in self.kernels:
                kernel.set_name(self.name)

            return

        self.cachekey = None
        self.liblang = [None, None]
        self.libpath = None 
        self.bldpath = None

        self.mapto      = pack_arguments(mapto)
        self.maptofrom  = pack_arguments(maptofrom)
        self.mapalloc  = pack_arguments(mapalloc)
        self.mapfrom    = pack_arguments(mapfrom)

        keys = [os.uname().release, version, self.name, self.kernels[0].compile]

        argindex = 0

        for maptype, data in zip(("to", "tofrom", "alloc", "from"),
                (self.mapto, self.maptofrom, self.mapalloc, self.mapfrom)):

            keys.append(maptype)

            for item in data:
                keys.append(item["data"].shape)
                keys.append(item["data"].dtype)

                item["index"] = argindex
                argindex += 1

        if self.liblang[0] is None or self.cache < MEMCACHE:
            lang, libpath, bldpath = self.build_sharedlib(gethash(str(keys)))
            self.liblang[0] = None
            self.liblang[1] = lang
            self.libpath = libpath
            self.bldpath = bldpath

        lib = None

        if self.liblang[0] is None:
            if self.libpath is not None and os.path.isfile(self.libpath):
                try:
                    lib = CDLL(self.libpath, mode=RTLD_GLOBAL)
                except:
                    pass

            if (lib is None and self.bldpath is not None and
                    os.path.isfile(self.bldpath)):
                lib = CDLL(self.bldpath, mode=RTLD_GLOBAL)

        if lib is not None:
            self.liblang[0] = lib
#
#        self.mapto      = pack_arguments(mapto)
#        self.maptofrom  = pack_arguments(maptofrom)
#        self.mapalloc  = pack_arguments(mapalloc)

        argtypes = []
        argitems = self.mapto+self.maptofrom+self.mapalloc+self.mapfrom

        for item in argitems:

            if self.liblang[1] == "cpp":
                flags = ["c_contiguous"]

            elif self.liblang[1] == "fortran":
                flags = ["f_contiguous"]

            else:
                raise Exception("Unknown language: %s" % self.liblang[1])

            argtypes.append(ndpointer(item["data"].dtype, flags=",".join(flags)))

        dataenter = getattr(self.liblang[0], "dataenter")
        dataenter.argtypes = argtypes

        dataenter(*[a["data"] for a in argitems])

        for kernel in self.kernels:
            kernel.set_name(self.name)

    def build_sharedlib(self, ckey):

        errmsgs = []

        compilers = get_compilers(self.name, self.lang,
                                    compile=self.kernels[0].compile,
                                    debug=self.debug)

        for comp in compilers:

            cachekey = ckey + "_" + comp.vendor + "".join(comp.version)

            if self.cache >= FILECACHE and cachekey in slib_cache:
                lang, basename, libext, libpath, bldpath = slib_cache[cachekey]
                self.cachekey = cachekey

                return lang, libpath, bldpath

            libdir = os.path.join(get_config("libdir"), comp.vendor, cachekey[:2])

            if not os.path.isdir(libdir):
                try:
                    os.makedirs(libdir)

                except FileExistsError:
                    pass

            basename = cachekey[2:]
            libname = basename + "." + comp.libext
            libpath = os.path.join(libdir, libname)

            if self.cache >= FILECACHE and os.path.isfile(libpath):
                self.cachekey = cachekey
                slib_cache[self.cachekey] = (comp.lang, basename, comp.libext,
                                                    libpath, None)
                return comp.lang, libpath, None

            try:
                codes, macros = self.gen_code(comp)

                if not os.path.isdir(get_config("blddir")):
                    set_config("blddir", tempfile.mkdtemp())

                bldpath = comp.compile(codes, macros)

                self.cachekey = cachekey
                slib_cache[self.cachekey] = (comp.lang, basename, comp.libext,
                                                    libdir, bldpath)
                return comp.lang, libpath, bldpath

            except Exception as err:
                errmsgs.append(str(err))

        raise Exception("\n".join(errmsgs))

    @abc.abstractmethod
    def get_dtype(self, arg):
        pass

    @abc.abstractmethod
    def gen_code(self, compiler):
        pass

    def wait(self, timeout=None):

        for kernel in self.kernels:
            kernel.wait(timeout=timeout)

        if self.offload:
            argtypes = []
            argitems = self.mapfrom

            for item in argitems:

                if self.liblang[1] == "cpp":
                    flags = ["c_contiguous"]

                elif self.liblang[1] == "fortran":
                    flags = ["f_contiguous"]

                else:
                    raise Exception("Unknown language: %s" % self.liblang[1])

                argtypes.append(ndpointer(item["data"].dtype, flags=",".join(flags)))

            dataexit = getattr(self.liblang[0], "dataexit")
            dataexit.argtypes = argtypes

            dataexit(*[a["data"] for a in argitems])

    def stop(self, timeout=None):

        self.wait(timeout=timeout)

        for kernel in self.kernels:
            kernel.stop(timeout=timeout)


def AccelData(*kernels, acctype=None, mapto=[], maptofrom=[], mapfrom=[],
                mapalloc=[], cache=MEMCACHE, profile=NOPROF, debug=NODEBUG,
                environ={}):

    if isinstance(acctype, str):
        acctype = (acctype,)

    accel = set(AccelDataBase.avails.keys()) if acctype is None else set(acctype)

    for kernel in kernels:
        kernel.spec.eval_pysection(environ)
        accel &= set(kernel.spec.list_sections())

    errmsgs = []

    for a in sorted(list(accel), key=(lambda x: accel_priority.index(x))):
        try:
            accel = AccelDataBase.avails[a](*kernels, mapto=mapto,
                            maptofrom=maptofrom, mapfrom=mapfrom,
                            mapalloc=mapalloc, cache=cache,
                            profile=profile, debug=debug)

            if debug > 0:
                print("DEBUG: using '%s' acceldata" % accel.name)

            return accel

        except Exception as err:
            errmsgs.append(repr(err))

    raise Exception("No acceldata is available: %s" % "\n".join(errmsgs))


