"""accelpy Accelerator module"""

import os, abc, time, threading, inspect
from ctypes import c_int, c_longlong, c_float, c_double, c_size_t
from numpy.ctypeslib import ndpointer
from numpy import ndarray, zeros, equal
from numpy.random import random
from collections import OrderedDict

from accelpy.core import Object
from accelpy.order import Order
from accelpy.compiler import Compiler
from accelpy import _config

LEN_TESTDATA = 10

class AccelBase(Object):
    """Accelerator Base Class"""

    # priority is implicit by loading subclasses at __init__.py
    avails = OrderedDict()

    dtypemap = {
        "int32": ["int", c_int],
        "int64": ["long", c_longlong],
        "float32": ["float", c_float],
        "float64": ["double", c_double]
    }

    _testdata = [
            {
                "data":random(LEN_TESTDATA),
                "curname": "accelpy_test_input"
            },
            {
                "data":zeros(LEN_TESTDATA),
                "curname": "accelpy_test_output"
            }
    ]


    def __init__(self, order, inputs, outputs, compilers=None):

        self._order = self._get_order(order)
        self._inputs, self._outputs = self._pack_arguments(inputs, outputs)

        self._sharedlib = self.build_sharedlib(compilers=compilers)

        self._copyin_cache = dict()
        self._copyout_cache = dict()
        self._malloc_cache = dict()

        self._time_start = None

    @abc.abstractmethod
    def gen_code(self, inputs, outputs):
        pass

    @abc.abstractmethod
    def getname_h2acopy(self, input):
        pass

    @abc.abstractmethod
    def getname_h2amalloc(self, input):
        pass

    @abc.abstractmethod
    def getname_a2hcopy(self, input):
        pass

    def get_argpair(self, arg):
        return (arg["data"].ndim, self.get_ctype(arg))

    def get_dtype(self, arg):
        return self.dtypemap[arg["data"].dtype.name][0]

    def get_vartype(self, arg, prefix=""):

        dtype = self.get_dtype(arg)
        return "%s%s_dim%d" % (prefix, dtype, arg["data"].ndim)

    def len_numpyattrs(self, arg):

        return 3 + len(arg["data"].shape)*2

    def get_numpyattrs(self, arg):
        data = arg["data"]

        return ((data.ndim, data.itemsize, data.size) + data.shape +
                tuple([int(s//data.itemsize) for s in data.strides]))

    def get_ctype(self, arg):

        return self.dtypemap[arg["data"].dtype.name][1]

    def _pack_arguments(self, inputs, outputs):

        def _tondarray(d):
            if isinstance(d, ndarray):
                return d

            elif isinstance(d, (list, tuple)):
                return asarray(d)

            else:
                raise Exception("No supported type: %s" % type(d))

        resin = []
        resout = []

        if isinstance(inputs, (tuple, list)):
            for input in inputs:
                inp = _tondarray(input)
                resin.append({"data": inp, "id": id(inp)})

        else:
            inp = _tondarray(input)
            resin = [{"data": inp, "id": id(inp)}]

        if isinstance(outputs, (tuple, list)):
            for output in outputs:
                outp = _tondarray(output)
                resout.append({"data": outp, "id": id(outp)})

        else:
            outp = _tondarray(outputs)
            resout = [{"data": outp, "id": id(outp)}]

        return resin, resout

    def _get_order(self, order):

        if isinstance(order, str):
            if os.path.isfile(order):
                with open(order) as fo:
                    order = Order(fo.read())
            else:
                order = Order(order)

        if not isinstance(order, Order):
            raise ("type '%s' is not valid order type." % type(order))

        return order

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
            return [_n(1), _n(1), _n(1)]

        # (members)
        elif len(workers) == 1:
            return [_n(1), _n(workers[0]), _n(1)]

        # (members, teams)
        elif len(workers) == 2:
            return [_n(workers[1]), _n(workers[0]), _n(1)]

        # (members, teams, assignments)
        elif len(workers) == 3:
            return [_n(workers[1]), _n(workers[0]), _n(workers[2])]

        else:
            raise Exception("Wrong # of worker initialization: %d" %
                        len(workers))

    def build_sharedlib(self, compilers=None):

        self._order.update_argnames(self._inputs, self._outputs)

        code = self.gen_code(self._inputs, self._outputs)

        if compilers is None:
            compilers = get_compilers(self.name)

        errmsgs = []

        for comp in compilers:
            try:
                lib = comp.compile(code)

                if self.h2a_func(self._testdata[0], "accelpy_test_h2acopy", lib=lib) != 0:
                    raise Exception("H2D copy test faild.")

                self._testdata[1]["data"].fill(0.0)

                if self.h2a_func(self._testdata[1], "accelpy_test_h2amalloc", lib=lib) != 0:
                    raise Exception("H2D malloc test faild.")

                testrun = getattr(lib, "accelpy_test_run")
                if testrun() != 0:
                    raise Exception("testrun faild.")

                if self.a2h_func(self._testdata[1], "accelpy_test_a2hcopy", lib=lib) != 0:
                    raise Exception("D2H copy test faild.")

                if not all(equal(self._testdata[1]["data"], self._testdata[1]["data"])):
                    raise Exception("data integrity check failure")

                return lib

            except Exception as err:
                errmsgs.append(str(err))

        raise Exception("\n".join(errmsgs))

    def h2a_func(self, arg, funcname, lib=None):

        if lib is None:
            lib = self._sharedlib

        attrs = self.get_numpyattrs(arg)
        cattrs = c_int*len(attrs)

        h2acopy = getattr(lib, funcname)
        h2acopy.restype = c_int
        h2acopy.argtypes = [ndpointer(self.get_ctype(arg)), cattrs, c_int]
        return h2acopy(arg["data"], cattrs(*attrs), len(attrs))

    def a2h_func(self, arg, funcname, lib=None):

        if lib is None:
            lib = self._sharedlib

        a2hcopy = getattr(lib, funcname)
        a2hcopy.restype = c_int
        a2hcopy.argtypes = [ndpointer(self.get_ctype(arg))]

        return a2hcopy(arg["data"])

    def _start_accel(self, device, channel, lib=None):

        if lib is None:
            lib = self._sharedlib

        start = getattr(lib, "accelpy_start")
        start.restype = c_int
        start.argtypes = [c_int, c_int]

        if start(device, channel) != 0:
            raise Exception("kernel run failed.")

    def run(self, *workers, device=0, channel=0):

        worker_triple = self._get_worker_triple(*workers)

        if self._inputs:
            for input in self._inputs:
                if input["id"] not in self._copyin_cache:
                    if self.h2a_func(input, self.getname_h2acopy(input)) != 0:
                        raise Exception("Accel h2a copy failed.")
                    self._copyin_cache[input["id"]] = input

        if self._outputs:
            for output in self._outputs:
                if output["id"] not in self._malloc_cache:
                    if self.h2a_func(output, self.getname_h2amalloc(output)) != 0:
                        raise Exception("Accel h2a malloc failed.")
                    self._malloc_cache[output["id"]] = output

        # run accel
        self._thread = threading.Thread(target=self._start_accel, args=(device, channel))
        self._thread.start()
        self._time_start = time.time()


    def wait(self, timeout=None, lib=None):

        if lib is None:
            lib = self._sharedlib

        stop = getattr(lib, "accelpy_stop")
        stop.restype = c_int
        stop.argtypes = []

        stop()

        if self._outputs:
            for output in self._outputs:
                if output["id"] not in self._copyout_cache:
                    if self.a2h_func(output, self.getname_a2hcopy(output)) != 0:
                        raise Exception("Accel a2h copy failed.")
                    self._copyout_cache[output["id"]] = output

        if timeout is None:
            self._thread.join()

        else:
            self._thread.join(max(0, timeout-time.time()+self._time_start))

class Accel(object):
    """Accelerator wrapper"""

    def __new__(cls, *vargs, **kwargs):

        kind = kwargs.pop("kind", None)

        if isinstance(kind, str):
            return AccelBase.avails[kind].__new__(cls, *vargs, **kwargs)

        elif kind is None or isinstance(kind, (list, tuple)):

            if kind is None:
                kind = AccelBasel.avails.keys()

            errmsgs = []

            for k in kind:
                try:
                    return AccelBase.avails[k].__new__(cls, *vargs, **kwargs)

                except Exception as err:
                    errmsgs.append(str(err))

            raise Exception("No accelerator is working: %s" % "\n".join(errmsgs))

        raise Exception("Kind '%s' is not valid." % str(kind))



def generate_compiler(compile):

    clist = compile.split()

    compcls = Compiler.from_path(clist[0])
    comp = compcls(option=" ".join(clist[1:]))

    return comp


def get_compilers(accel, compiler=None):
    """
        parameters:

        accel: the accelerator id or a list of them
        compiler: compiler path or compiler id, or a list of them 
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

    if compiler:
        if isinstance(compiler, str):
            citems = compiler.split()

            if not citems:
                raise Exception("Blank compiler")

            if os.path.isfile(citems[0]):
                compilers.append(generate_compiler(compiler))

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

        elif isinstance(compiler, Compiler):
            compilers.append(compiler)

        elif isinstance(compiler (list, tuple)):

            for comp in compiler:
                if isinstance(comp, str):
                    citems = comp.split()

                    if not citems:
                        raise Exception("Blank compiler")

                    if os.path.isfile(citems[0]):
                        try:
                            compilers.append(generate_compiler(comp))
                        except:
                            pass

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

                elif isinstance(comp, Compiler):
                    compilers.append(comp)

                else:
                    raise Exception("Unsupported compiler type: %s" % str(comp))
        else:
            raise Exception("Unsupported compiler type: %s" % str(compiler))

    new_compilers = []
    errmsgs = []

    if compilers:
        for comp in compiers:
            if any(comp.accel==a[0] and comp.lang==a[1] for a in accels):
                new_compilers.append(comp)
    else:
        for acc, lang in accels:
            vendors = Compiler.avails[lang][acc]
            for vendor, cls in vendors.items():
                try:
                    new_compilers.append(cls())
                except Exception as err:
                    errmsgs.append(str(err))

    if not new_compilers:
        print("\n".join(errmsgs))

    return new_compilers
