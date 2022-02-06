"""accelpy Kernel module"""


import os, abc, time, threading, shutil, numpy, tempfile

from ctypes import c_int64
from numpy.ctypeslib import ndpointer, load_library
from collections import OrderedDict

from accelpy.const import version
from accelpy.util import Object, gethash, get_config, set_config
from accelpy.spec import Spec
from accelpy.compiler import Compiler
from accelpy import cache


class Task(Object):

    def __init__(self, lang, libpath, bldpath):

        self.lib = None
        self.lang = lang
        self.libpath = libpath
        self.bldpath = bldpath
        self.synched = False

    def run(self, data, getname):

        self.thread = threading.Thread(target=self._start_kernel,
                            args=(data, getname))
        self.start = time.time()
        self.thread.start()

    def varmap(self, arg, funcname):

        if self.lang == "cpp":
            flags = ["c_contiguous"]

        elif self.lang == "fortran":
            flags = ["f_contiguous"]

        else:
            raise Exception("Unknown language: %s" % lang)

        # TODO consider to use "numpy.ascontiguousarray"

        datamap = getattr(self.lib, funcname)
        datamap.restype = c_int64
        datamap.argtypes = [ndpointer(arg["data"].dtype, flags=",".join(flags))]

        res = datamap(arg["data"])

        return res

    def _start_kernel(self, data, getname):

        if self.libpath is not None and os.path.isfile(self.libpath):
            try:
                libdir, libname = os.path.split(self.libpath)
                basename, _ = os.path.splitext(libname)

                self.lib = load_library(basename, libdir)
            except:
                pass

        if self.lib is None and self.bldpath is not None and os.path.isfile(self.bldpath):
            blddir, bldname = os.path.split(self.bldpath)
            basename, _ = os.path.splitext(bldname)

            self.lib = load_library(basename, blddir)

        for arg in data:
            self.varmap(arg, getname(arg))

        start = getattr(self.lib, "accelpy_start")
        start.restype = c_int64
        start.argtypes = []

        res = start()

        if res != 0:
            raise Exception("kernel returns non-zero: %d" % res)

    def wait(self, timeout=None):

        timeout = max(0, self.start+timeout-time.time()) if timeout else None
        self.thread.join(timeout=timeout)
        self.thread.join()

        if not os.path.isfile(self.libpath) and os.path.isfile(self.bldpath):
            try:
                shutil.copyfile(self.bldpath, self.libpath)

            except FileExistsError:
                pass

        self.synched = True

    def stop(self, timeout=None):

        if not self.synched:
            self.wait(timeout=timeout)

        accstop = getattr(self.lib, "accelpy_stop")
        accstop.restype = c_int64
        accstop.argtypes = []

        res = accstop()

        return res


class KernelBase(Object):

    avails = OrderedDict()

    def __init__(self, spec, compile=None, debug=False):

        self.debug = debug
        self.spec = spec
        self.compile = compile
        self.cachekey = None
        self.lib = None

        if self.spec is None:
            raise Exception("No kernel spec is found")

        self.tasks = []

    def launch(self, *data, specenv={}, reload=False):

        self.spec.eval_pysection(specenv)
        self.section = self.spec.get_section(self.name)

        self.data = self._pack_arguments(data)
        self.spec.update_argnames(self.data)
        self.section.update_argnames(self.data)

        keys = [os.uname().release, version, self.name,
                self.compile, self.section.hash()]

        for item in self.data:
            keys.append(item["data"].shape)
            keys.append(item["data"].dtype)

        lang, libpath, bldpath = self.build_sharedlib(gethash(str(keys)), reload)

        task = Task(lang, libpath, bldpath)
        task.run(self.data, self.getname_varmap)

        self.tasks.append(task)

        return task

    def wait(self, *tasks, timeout=None):

        if len(tasks) > 0:
            for task in tasks:
                task.wait(timeout=timeout)

        else:
            for task in self.tasks:
                task.wait(timeout=timeout)

    def stop(self, timeout=None):

        for task in self.tasks:
            if not task.synched:
                task.wait(timeout=timeout)

            task.stop(timeout=timeout)

    def get_dtype(self, arg):
        return self.dtypemap[arg["data"].dtype.name][0]

    @abc.abstractmethod
    def getname_varmap(self, arg):
        pass

    def _pack_arguments(self, data):

        res = []

        for arg in data:
            idarg = id(arg)

            if isinstance(arg, numpy.ndarray):
                res.append({"data": arg, "id": idarg, "curname": None})

            else:
                newarg = numpy.asarray(arg)
                res.append({"data": newarg, "id": idarg, "curname": None,
                            "orgdata": arg})

        return res

    def get_include(self):
        return ""

    def add_includes(self):

        incs = []

        # TODO: implement this
        #import pdb; pdb.set_trace()

        return incs

    def build_sharedlib(self, ckey, reload):

        errmsgs = []

        compilers = get_compilers(self.name, compile=self.compile)

        for comp in compilers:

            cachekey = ckey + "_" + comp.vendor

            if not reload and cachekey in cache.sharedlib:
                lang, basename, libext, libdir, bldpath = cache.sharedlib[cachekey]
                libpath = os.path.join(libdir, basename+"."+libext)
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

            try:
                codes, macros = self.gen_code(comp)

                if not os.path.isdir(get_config("blddir")):
                    set_config("blddir", tempfile.mkdtemp())

                bldpath = comp.compile(codes, macros, self.debug)

                cache.sharedlib[self.cachekey] = (comp.lang, basename, comp.libext,
                                                    libdir, bldpath)
                return comp.lang, libpath, bldpath

            except Exception as err:
                errmsgs.append(str(err))

        raise Exception("\n".join(errmsgs))

    @abc.abstractmethod
    def gen_code(self, compiler):
        pass


def Kernel(spec, accel=None, compile=None, debug=False):

    if isinstance(accel, str):
        return KernelBase.avails[accel](spec, compile=compile, debug=debug)

    elif accel is None or isinstance(accel, (list, tuple)):

        if accel is None:
            accel = KernelBase.avails.keys()

        errmsgs = []

        for k in accel:
            try:
                kernel = KernelBase.avails[k](spec, compile=compile, debug=debug)
                return kernel

            except Exception as err:
                errmsgs.append(repr(err))

        raise Exception("No kernel is working: %s" % "\n".join(errmsgs))

    raise Exception("Accelerator '%s' is not valid." % str(accel))


def generate_compiler(compile):

    clist = compile.split()

    compcls = Compiler.from_path(clist[0])
    comp = compcls(option=" ".join(clist[1:]))

    return comp


def get_compilers(kernel, compile=None):
    """
        parameters:

        kernel: the kernel id or a list of them
        compile: compiler path or compiler id, or a list of them
                  syntax: "vendor|path [{default}] [additional options]"
"""

    kernels = []

    if isinstance(kernel, str):
        kernels.append((kernel, KernelBase.avails[kernel].lang))

    elif isinstance(kernel, KernelBase):
        kernels.append((kernel.name, kernel.lang))

    elif isinstance(kernel, (list, tuple)):
        for k in kernel:
            if isinstance(k, str):
                kernels.append((k, KernelBase.avails[k].lang))

            elif isinstance(k, KernelBase):
                kernels.append((k.name, k.lang))

            else:
                raise Exception("Unknown kernel type: %s" % str(k))

    else:
        raise Exception("Unknown kernel type: %s" % str(kernel))

    compilers = []

    if compile:
        if isinstance(compile, str):
            citems = compile.split()

            if not citems:
                raise Exception("Blank compile")

            if os.path.isfile(citems[0]):
                compilers.append(generate_compiler(compile))

            else:
                # TODO: vendor name search
                for lang, langsubc in Compiler.avails.items():
                    for kernel, kernelsubc in langsubc.items():
                        for vendor, vendorsubc in kernelsubc.items():
                            if vendor == citems[0]:
                                try:
                                    compilers.append(vendorsubc(option=" ".join(citems[1:])))
                                except:
                                    pass

        elif isinstance(compile, Compiler):
            compilers.append(compile)

        elif isinstance(compile, (list, tuple)):

            for comp in compile:
                if isinstance(comp, str):
                    citems = comp.split()

                    if not citems:
                        raise Exception("Blank compile")

                    if os.path.isfile(citems[0]):
                        try:
                            compilers.append(generate_compiler(comp))
                        except:
                            pass

                    else:
                        # TODO: vendor name search
                        for lang, langsubc in Compiler.avails.items():
                            for kernel, kernelsubc in langsubc.items():
                                for vendor, vendorsubc in kernelsubc.items():
                                    if vendor == citems[0]:
                                        try:
                                            compilers.append(vendorsubc(option=" ".join(citems[1:])))
                                        except:
                                            pass

                elif isinstance(comp, Compiler):
                    compilers.append(comp)

                else:
                    raise Exception("Unsupported compiler type: %s" % str(comp))
        else:
            raise Exception("Unsupported compiler type: %s" % str(compile))

    return_compilers = []
    errmsgs = []


    if compilers:
        for comp in compilers:
            if any(comp.accel==k[0] and comp.lang==k[1] for k in kernels):
                return_compilers.append(comp)

    elif compile is None:
        for acc, lang in kernels:

            if lang not in Compiler.avails:
                continue

            if acc not in Compiler.avails[lang]:
                continue

            vendors = Compiler.avails[lang][acc]

            for vendor, cls in vendors.items():
                try:
                    return_compilers.append(cls())
                except Exception as err:
                    errmsgs.append(str(err))


    if not return_compilers:
        if errmsgs:
            raise Exception("No compiler is found: %s" % "\n".join(errmsgs))
        else:
            raise Exception("No compiler is found: %s" % str(kernels))

    return return_compilers

