"""accelpy Accelerator module"""

import os, sys, abc, time, threading, inspect, numpy, hashlib, shutil, tempfile
from ctypes import c_int64, POINTER, byref
from numpy.ctypeslib import ndpointer, load_library
from numpy.random import random
from collections import OrderedDict

from accelpy.core import Object, version
from accelpy.order import Order
from accelpy.compiler import Compiler
from accelpy import _config

LEN_TESTDATA = 10

_cache = {
    "sharedlib": {},
    "test": {}
}

class AccelBase(Object):
    """Accelerator Base Class"""

    # priority is implicit by loading subclasses at __init__.py
    avails = OrderedDict()

    _testdata = [
            {
                "data":random(LEN_TESTDATA),
                "curname": "accelpy_test_input"
            },
            {
                "data":numpy.zeros(LEN_TESTDATA),
                "curname": "accelpy_test_output"
            }
    ]


    def __init__(self, *vargs, kind=None, compile=None, debug=False):

        self._debug = debug
        self._order = None

        inputs, outputs = [], []
        for varg in vargs:
            if isinstance(varg, Order):
                self._order = varg

            elif self._order is None:
                inputs.append(varg)

            else:
                outputs.append(varg)

        self._inputs, self._outputs = self._pack_arguments(inputs, outputs)
        self._order.update_argnames(self._inputs, self._outputs)
        self._ordersec =  self._order.get_section(self.name)
        self._orderhash =  hashlib.md5("".join(self._ordersec.body).encode("utf-8")
                            ).hexdigest()[:10]
        self._compile = compile
        self._threads_run = {} # run_id: [thread, state(0:started 1:stopped), start time, sharedlib]

    @abc.abstractmethod
    def gen_code(self, compiler, inputs, outputs, worker_triple, run_id, device, channel):
        pass

    @abc.abstractmethod
    def getname_h2acopy(self, arg):
        pass

    @abc.abstractmethod
    def getname_h2amalloc(self, arg):
        pass

    @abc.abstractmethod
    def getname_a2hcopy(self, arg):
        pass

    def get_argpair(self, arg):
        return (arg["data"].ndim, self.get_ctype(arg))

    def get_dtype(self, arg):
        return self.dtypemap[arg["data"].dtype.name][0]

    def get_ctype(self, arg):

        return self.dtypemap[arg["data"].dtype.name][1]

    def get_shapestr(self, arg):
        return ",".join([str(s) for s in arg["data"].shape])

    def get_stridestr(self, arg):
        return ",".join([str(int(s//arg["data"].itemsize)) for s
                in arg["data"].strides])

    def _pack_arguments(self, inputs, outputs):

        resin = []
        resout = []

        if inputs is None:
            inputs = []

        if outputs is None:
            outputs = []

        for inp in inputs:
            if isinstance(inp, numpy.ndarray):
                resin.append({"data": inp, "id": id(inp), "h2acopy": False})

            else:
                newinp = numpy.asarray(inp)
                resin.append({"data": newinp, "id": id(newinp), "h2acopy": False,
                            "orgdata": inp})

        for outp in outputs:
            if isinstance(outp, numpy.ndarray):
                resout.append({"data": outp, "id": id(outp), "h2acopy": False})
            else:
                newoutp = numpy.asarray(outp)
                resout.append({"data": newoutp, "id": id(newoutp), "h2acopy": False,
                            "orgdata": outp})

        return resin, resout

    def _get_worker_triple(self, *workers):

        def _n(s):

            if isinstance(s, int):
                return ((s,1,1))

            elif isinstance(s, (list, tuple)):
                l = len(s)
                if l >= 3:
                    return tuple(s[:3])

                elif l == 2:
                    return (s[0], s[1], 1)

                elif l == 1:
                    return (s[0], 1, 1)

                else:
                    raise Exception("Wrong number of worker dimension: %d" % l)
            else:
                raise Exception("Unsupported size type: %s" % str(s))

        # ()
        if len(workers) == 0:
            return (_n(1), _n(1), _n(1))

        # (members)
        elif len(workers) == 1:
            return (_n(1), _n(workers[0]), _n(1))

        # (members, teams)
        elif len(workers) == 2:
            return (_n(workers[1]), _n(workers[0]), _n(1))

        # (members, teams, assignments)
        elif len(workers) == 3:
            return (_n(workers[1]), _n(workers[0]), _n(workers[2]))

        else:
            raise Exception("Wrong # of worker initialization: %d" %
                        len(workers))

    def build_sharedlib(self, run_id, device, channel, wtriple, cachekey):

        errmsgs = []
        compilers = get_compilers(self.name, compile=self._compile)

        if not os.path.isdir(_config["blddir"]):
            _config["blddir"] = tempfile.mkdtemp()

        for comp in compilers:
            cachedir = os.path.join(_config["libdir"], comp.vendor, cachekey[:2])
            cachelib = os.path.join(cachedir, cachekey[2:]+"."+comp.libext)

            try:
                os.makedirs(cachedir)

            except FileExistsError:
                pass

            if os.path.isfile(cachelib):

                libdir, libname = os.path.split(cachelib)
                name, _ = os.path.splitext(libname)

                lib = load_library(name, libdir)

                if lib is None:
                    continue

                return lib, comp.lang

            try:
                code, macros = self.gen_code(comp, self._inputs, self._outputs,
                                    wtriple, run_id, device, channel)

                macros["ACCELPY_ACCEL_RUNID"] = run_id
                macros["ACCELPY_ACCEL_DEVICE"] = device
                macros["ACCELPY_ACCEL_CHANNEL"] = channel
                macros["ACCELPY_TEAM_DIM0"] = wtriple[0][0]
                macros["ACCELPY_TEAM_DIM1"] = wtriple[0][1]
                macros["ACCELPY_TEAM_DIM2"] = wtriple[0][2]
                macros["ACCELPY_WORKER_DIM0"] = wtriple[1][0]
                macros["ACCELPY_WORKER_DIM1"] = wtriple[1][1]
                macros["ACCELPY_WORKER_DIM2"] = wtriple[1][2]
                macros["ACCELPY_ASSIGN_DIM0"] = wtriple[2][0]
                macros["ACCELPY_ASSIGN_DIM1"] = wtriple[2][1]
                macros["ACCELPY_ASSIGN_DIM2"] = wtriple[2][2]

                libfile = comp.compile(code, macros, self._debug)

                libdir, libname = os.path.split(libfile)
                name, _ = os.path.splitext(libname)

                lib = load_library(name, libdir)

                if lib is None:
                    continue

                if self.h2acopy(lib, self._testdata[0], "accelpy_test_h2acopy") != 0:
                    raise Exception("H2D copy test faild.")

                self._testdata[1]["data"].fill(0.0)

                if self.h2amalloc(lib, self._testdata[1], "accelpy_test_h2amalloc",
                        writable=True) != 0:
                    raise Exception("H2D malloc test faild.")

                testrun = getattr(lib, "accelpy_test_run")
                if testrun() != 0:
                    raise Exception("testrun faild.")

                if self.a2hcopy(lib, self._testdata[1], "accelpy_test_a2hcopy") != 0:
                    raise Exception("D2H copy test faild.")

                if not all(numpy.equal(self._testdata[0]["data"], self._testdata[1]["data"])):
                    raise Exception("accel test result mismatch: %s != %s" %
                        (str(self._testdata[0]["data"]), str(self._testdata[1]["data"])))

                if not os.path.isfile(cachelib):
                    shutil.copyfile(libfile, cachelib)
                    _cache["sharedlib"][cachekey] = (lib, comp.lang)

                return lib, comp.lang

            except Exception as err:
                errmsgs.append(str(err))
#
#        else: # not master
#            TIMEOUT = 20
#            start = time.time()
#
#            while time.time() - start < TIMEOUT:
#                for comp in compilers:
#
#                    cachedir = os.path.join(_config["libdir"], comp.vendor, cachekey[:2])
#                    cachelib = os.path.join(cachedir, cachekey[2:]+"."+comp.libext)
#
#                    if os.path.isfile(cachelib):
#
#                        libdir, libname = os.path.split(cachelib)
#                        name, _ = os.path.splitext(libname)
#
#                        lib = load_library(name, libdir)
#
#                        if lib is None:
#                            continue
#                        
#                        _cache["sharedlib"][cachekey] = (lib, comp.lang)
#                        return _cache["sharedlib"][cachekey]
#
#                time.sleep(0.1)

        raise Exception("\n".join(errmsgs))

    def _datacopy(self, lib, arg, funcname, writable=None):

        if lib is None:
            return -1

        if self.lang == "cpp":
            flags = ["c_contiguous"]

        elif self.lang == "fortran":
            flags = ["f_contiguous"]

        else:
            raise Exception("Unknown language: %s" % lang)

        if writable:
            flags.append("writeable")

        datacopy = getattr(lib, funcname)
        datacopy.restype = c_int64
        datacopy.argtypes = [ndpointer(arg["data"].dtype, flags=",".join(flags))]

        res = datacopy(arg["data"])

        return res

    def h2acopy(self, lib, arg, funcname, writable=None):

        res = self._datacopy(lib, arg, funcname, writable=writable)

        if res == 0:
            arg["h2acopy"] = True

        return res

    def h2amalloc(self, lib, arg, funcname, writable=None):

        res = self._datacopy(lib, arg, funcname, writable=writable)

        if res == 0:
            arg["h2amalloc"] = True

        return res

    def a2hcopy(self, lib, arg, funcname, writable=None):

        res = self._datacopy(lib, arg, funcname, writable=writable)

        if res == 0:
            arg["a2hcopy"] = True

        if "orgdata" in arg:
            for i, d in enumerate(arg["data"]):
                arg["orgdata"][i] = d

        return res

    def _start_accel(self, run_id, device, channel, workers, inputs, outputs):

        # TODO: workers in a team directly share resources such as memory or io. A worker is corresponding to a thread in openmp or worker in openacc
        # TODO: workers in a different team need some additional work to share a resouce. A team is correspondng to process or gang in openacc or block in cuda
        # TODO: if an accel does not support those concept ( for example, openmp does not have a concept of process within) then the setting silently ignored
        # TODO: if workers passes the max limit of the device, the exact value might be silently adjusted.
        # TODO: unused concpet will be provided to the order code through possibly macros
        # TODO: Use compiler macros instead of passing values to support workers, teams, and assignments
        # compilation should be done first

        worker_triple = self._get_worker_triple(*workers)

        inputs, outputs = self._pack_arguments(inputs, outputs)
        self._order.update_argnames(inputs, outputs)

        _inputs = inputs if inputs else self._inputs
        _outputs = outputs if outputs else self._outputs

        keys = [os.uname().release, version, self.name, str(self._compile),
                    self._orderhash, device, channel, worker_triple]

        for item in _inputs+_outputs:
            keys.append(item["data"].shape)
            keys.append(item["data"].dtype)

        cachekey = hashlib.md5(str(keys).encode("utf-8")).hexdigest()[:10]

        #lib, self.lang = self.build_sharedlib(run_id, device, channel, worker_triple)

        if cachekey in _cache["sharedlib"]:
            lib, self.lang = _cache["sharedlib"][cachekey]

        else:
            _cache["sharedlib"][cachekey] = self.build_sharedlib(run_id,
                                            device, channel, worker_triple, cachekey)
            lib, self.lang = _cache["sharedlib"][cachekey]

        if lib is None:
            raise Exception("Can not build shared library")

        self._threads_run[run_id][3] = lib 

        if _inputs:
            for input in _inputs:
                if "h2acopy" not in input or not input["h2acopy"]:
                    if self.h2acopy(lib, input, self.getname_h2acopy(input)) != 0:
                        raise Exception("Accel h2a copy failed.")

        if _outputs:
            for output in _outputs:
                if "h2amalloc" not in output or not output["h2amalloc"]:
                    if self.h2amalloc(lib, output, self.getname_h2amalloc(output), writable=True) != 0:
                        raise Exception("Accel h2a malloc failed.")

        start = getattr(lib, "accelpy_start")
        start.restype = c_int64
        start.argtypes = []

        if start() != 0:
            raise Exception("kernel run failed.")

    def run(self, *workers, device=0, channel=0, timeout=None, inputs=None, outputs=None):

        # run accel
        run_id = len(self._threads_run)

        # TODO: use lock to support multiple accelerators uses device simultaneously.

        thread = threading.Thread(target=self._start_accel,
                            args=(run_id, device, channel, workers, inputs, outputs))
        self._threads_run[run_id] = [thread, 0, time.time(), None]
        thread.start()

        return run_id

    def output(self, run_id=0, output=None, force=False, timeout=None):

        self.wait(run_id=run_id, timeout=timeout)

        if run_id is None:
            run_ids = self._threads_run.keys()

        elif isinstance(run_id, int):
            run_ids = [run_id]

        else:
            run_ids = run_id

        for run_id in run_ids:

            tobj, state, stime, slib = self._threads_run[run_id]

            outputs = output if output else self._outputs

            for outp in outputs:
                if "a2hcopy" not in outp or not outp["a2hcopy"] or force:
                    if self.a2hcopy(slib, outp, self.getname_a2hcopy(outp)) != 0:
                        raise Exception("Accel a2h copy failed.")

    def stop(self, run_id=None, output=True, timeout=None):

        self.wait(run_id=run_id, timeout=timeout)

        if run_id is None:
            run_ids = self._threads_run.keys()

        elif isinstance(run_id, int):
            run_ids = [run_id]

        else:
            run_ids = run_id


        for run_id in run_ids:

            tobj, state, stime, slib = self._threads_run[run_id]

            if state == 1:
                continue

            self.wait(run_id=run_id, timeout=timeout)

            if output is True:
                self.output()

            elif output:
                self.output(output=output)

            stop = getattr(slib, "accelpy_stop")
            stop.restype = c_int64
            stop.argtypes = []

            stop()

    def wait(self, run_id=None, timeout=None):

        if run_id is None:
            run_ids = self._threads_run.keys()

        elif isinstance(run_id, int):
            run_ids = [run_id]

        else:
            run_ids = run_id

        for run_id in run_ids:

            tobj, state, stime, slib = self._threads_run[run_id]

            timeout = max(0, stime+timeout-time.time()) if timeout else None
            tobj.join(timeout=timeout)


def Accel(*vargs, **kwargs):

    kind = kwargs.pop("kind", None)

    order = None
    for varg in vargs:
        if isinstance(varg, Order):
            order = varg
            break

    if order is None:
        raise Exception("No order is found")

    if isinstance(kind, str):
        return AccelBase.avails[kind](*vargs, **kwargs)

    elif kind is None or isinstance(kind, (list, tuple)):

        if kind is None:
            kind = AccelBase.avails.keys()

        errmsgs = []

        for k in kind:
            try:
                order.get_section(k)
                accel = AccelBase.avails[k](*vargs, **kwargs)
                return accel 

            except Exception as err:
                errmsgs.append(str(err))

        raise Exception("No accelerator is working: %s" % "\n".join(errmsgs))

    raise Exception("Kind '%s' is not valid." % str(kind))


def generate_compiler(compile):

    clist = compile.split()

    compcls = Compiler.from_path(clist[0])
    comp = compcls(option=" ".join(clist[1:]))

    return comp


def get_compilers(accel, compile=None):
    """
        parameters:

        accel: the accelerator id or a list of them
        compile: compiler path or compiler id, or a list of them 
                  syntax: "vendor|path [{default}] [additional options]"
"""

    accels = []

    if isinstance(accel, str):
        accels.append((accel, AccelBase.avails[accel].lang))

    elif isinstance(accel, AccelBase):
        accels.append((accel.name, accel.lang))

    elif isinstance(accel, (list, tuple)):
        for a in accel:
            if isinstance(a, str):
                accels.append((a, AccelBase.avails[a].lang))

            elif isinstance(a, AccelBase):
                accels.append((a.name, a.lang))

            else:
                raise Exception("Unknown accelerator type: %s" % str(a))

    else:
        raise Exception("Unknown accelerator type: %s" % str(accel))

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
                for lang, langsubc in Compiler.avalis.items():
                    for accel, accelsubc in langsubc.items():
                        for vendor, vendorsubc in accelsubc.items():
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
                            for accel, accelsubc in langsubc.items():
                                for vendor, vendorsubc in accelsubc.items():
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
            if any(comp.accel==a[0] and comp.lang==a[1] for a in accels):
                return_compilers.append(comp)

    elif compile is None:
        for acc, lang in accels:

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
            raise Exception("No compiler is found: %s" % str(accels))

    return return_compilers
