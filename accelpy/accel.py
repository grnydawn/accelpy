"""accelpy Accelerator module"""

import os, abc, time, threading, inspect, numpy
from ctypes import c_int64, POINTER, byref
from numpy.ctypeslib import ndpointer
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


    def __init__(self, order, inputs, outputs, compile=None):

        self._order = self._get_order(order)
        self._inputs, self._outputs = self._pack_arguments(inputs, outputs)

        self._order.update_argnames(self._inputs, self._outputs)

        self._sharedlib = None
        self._threads_run = []
        self._thread_compile = threading.Thread(target=self._build_sharedlib, args=(compile,))
        self._thread_compile.start()

        self._stopped = False
        self._time_start = time.time()

    def _build_sharedlib(self, compile):

        self._sharedlib = self.build_sharedlib(compile=compile)


    @abc.abstractmethod
    def gen_code(self, inputs, outputs):
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

    #def len_numpyattrs(self, arg):

    #    return 3 + len(arg["data"].shape)*2

    def get_numpyattrs(self, arg):
        data = arg["data"]

        return numpy.array((data.size, data.ndim, data.itemsize) + data.shape +
                tuple([int(s//data.itemsize) for s in data.strides]), dtype=numpy.int64)

    def get_ctype(self, arg):

        return self.dtypemap[arg["data"].dtype.name][1]

    def _pack_arguments(self, inputs, outputs):

        def _tondarray(d):
            if isinstance(d, numpy.ndarray):
                return d

            elif isinstance(d, (list, tuple)):
                return numpy.asarray(d)

            else:
                raise Exception("No supported type: %s" % type(d))

        resin = []
        resout = []

        if isinstance(inputs, (tuple, list)):
            for input in inputs:
                inp = _tondarray(input)
                resin.append({"data": inp, "id": id(inp), "h2acopy": False})

        else:
            inp = _tondarray(input)
            resin = [{"data": inp, "id": id(inp), "h2acopy": False}]

        if isinstance(outputs, (tuple, list)):
            for output in outputs:
                outp = _tondarray(output)
                resout.append({"data": outp, "id": id(outp), "a2hcopy": False})

        else:
            outp = _tondarray(outputs)
            resout = [{"data": outp, "id": id(outp), "a2hcopy": False}]

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

    def build_sharedlib(self, compile=None):

        code = self.gen_code(self._inputs, self._outputs)

        compilers = get_compilers(self.name, compile=compile)

        errmsgs = []

        for comp in compilers:
            try:
                lib = comp.compile(code)

                if lib is None:
                    continue

                if self.h2acopy(self._testdata[0], "accelpy_test_h2acopy", lib=lib) != 0:
                    raise Exception("H2D copy test faild.")

                self._testdata[1]["data"].fill(0.0)

                if self.h2amalloc(self._testdata[1], "accelpy_test_h2amalloc",
                        lib=lib, writable=True) != 0:
                    raise Exception("H2D malloc test faild.")

                testrun = getattr(lib, "accelpy_test_run")
                if testrun() != 0:
                    raise Exception("testrun faild.")

                if self.a2hcopy(self._testdata[1], "accelpy_test_a2hcopy", lib=lib) != 0:
                    raise Exception("D2H copy test faild.")

                if not all(numpy.equal(self._testdata[0]["data"], self._testdata[1]["data"])):
                    raise Exception("data integrity check failure")

                return lib

            except Exception as err:
                errmsgs.append(str(err))

        raise Exception("\n".join(errmsgs))

    def _datacopy(self, arg, funcname, lib=None, writable=None):

        if lib is None:
            lib = self._sharedlib

        if lib is None:
            return -1

        flags = ["contiguous"]

        if writable:
            flags.append("writeable")

        attrs = self.get_numpyattrs(arg)

        datacopy = getattr(lib, funcname)
        datacopy.restype = c_int64
        datacopy.argtypes = [
            POINTER(c_int64),
            ndpointer(attrs.dtype, flags=",".join(flags)),
            ndpointer(arg["data"].dtype, flags=",".join(flags))
            ]
        res = datacopy(byref(c_int64(attrs.size)), attrs, arg["data"])

        return res

    def h2acopy(self, arg, funcname, lib=None, writable=None):
        res = self._datacopy(arg, funcname, lib=lib, writable=writable)

        if res == 0:
            arg["h2acopy"] = True

        return res

    def h2amalloc(self, arg, funcname, lib=None, writable=None):
        res = self._datacopy(arg, funcname, lib=lib, writable=writable)

        if res == 0:
            arg["h2amalloc"] = True

        return res

    def a2hcopy(self, arg, funcname, lib=None, writable=None):
        res = self._datacopy(arg, funcname, lib=lib, writable=writable)

        if res == 0:
            arg["a2hcopy"] = True

        return res

    def _start_accel(self, run_id, device, channel, workers, lib=None):

        if lib is None:
            lib = self._sharedlib

        start = getattr(lib, "accelpy_start")
        start.restype = c_int64
        start.argtypes = [POINTER(c_int64)]*12

        args = [byref(c_int64(run_id)), byref(c_int64(device)),
                byref(c_int64(channel))]

        for (x, y, z) in workers:
            args.append(byref(c_int64(x)))
            args.append(byref(c_int64(y)))
            args.append(byref(c_int64(z)))

        if start(*args) != 0:
            raise Exception("kernel run failed.")

    def run(self, *workers, device=0, channel=0, timeout=None, inputs=None, outputs=None):

        # TODO: keep allocated address per data movements in accel

        # compilation should be done first
        self._thread_compile.join()

        worker_triple = self._get_worker_triple(*workers)

        _inputs = inputs if inputs else self._inputs

        if _inputs:
            for input in _inputs:
                if "h2acopy" not in input or not input["h2acopy"]:
                    if self.h2acopy(input, self.getname_h2acopy(input)) != 0:
                        raise Exception("Accel h2a copy failed.")

        _outputs = outputs if outputs else self._outputs

        if _outputs:
            for output in _outputs:
                if "h2amalloc" not in output or not output["h2amalloc"]:
                    if self.h2amalloc(output, self.getname_h2amalloc(output), writable=True) != 0:
                        raise Exception("Accel h2a malloc failed.")

        # run accel
        run_id = len(self._threads_run)

        thread_run = threading.Thread(target=self._start_accel,
                            args=(run_id, device, channel, worker_triple))
        thread_run.start()
        self._threads_run.append((thread_run, time.time()))

        return run_id

    def output(self, output=None, force=False):

        outputs = output if output else self._outputs

        for outp in self._outputs:
            if "a2hcopy" not in outp or not outp["a2hcopy"] or force:
                if self.a2hcopy(outp, self.getname_a2hcopy(outp)) != 0:
                    raise Exception("Accel a2h copy failed.")

    def stop(self, lib=None, output=True, timeout=None):

        # TODO: deallocated the addresses per data movements in accel

        if self._stopped:
            return

        self.wait(timeout=timeout)

        if lib is None:
            lib = self._sharedlib

        stop = getattr(lib, "accelpy_stop")
        stop.restype = c_int64
        stop.argtypes = []

        stop()

        self._stopped = True

        if output is True:
            self.output()

        elif output:
            self.output(output=output)

    def wait(self, run_id=None, timeout=None):

        self._thread_compile.join()

        for rid, (thread_run, time_start) in enumerate(self._threads_run):
            if run_id:
                if isinstance(run_id, int):
                    if rid != run_id: continue

                elif isinstance(int, (list, tuple)):
                    for r in run_id:
                        if rid != r: contine

            if thread_run:
                timeout = max(0, time_start+timeout-time.time()) if timeout else None
                thread_run.join(timeout=timeout)


def Accel(*vargs, **kwargs):

    kind = kwargs.pop("kind", None)

    if isinstance(kind, str):
        return AccelBase.avails[kind](*vargs, **kwargs)

    elif kind is None or isinstance(kind, (list, tuple)):

        if kind is None:
            kind = AccelBasel.avails.keys()

        errmsgs = []

        for k in kind:
            try:
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
