"""accelpy Accelerator module"""

import os, abc, time, threading, inspect
#from numpy.ctypeslib import load_library
from numpy import ndarray
from collections import OrderedDict

from accelpy.core import Object
from accelpy.order import Order
from accelpy import _config


class AccelBase(Object):
    """Accelerator Base Class"""

    # priority is implicit by loading subclasses at __init__.py
    avails = OrderedDict()

    def __init__(self, order, inputs, outputs, compilers=None):

        self._order = self._get_order(order)
        self._inputs, self._outputs = self._pack_arguments(inputs, outputs)

        self._sharedlib = self.build_sharedlib(compilers=compilers)

        self._time_start = None

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

    @abc.abstractmethod
    def build_sharedlib(self, compilers=None):
        pass

    def _start_accel(self, device, channel):
        # ctype
        self._sharedlib.run(device, channel)

    def run(self, *workers, device=0, channel=0):

        worker_triple = self._get_worker_triple(*workers)

        if self._inputs:
            for input in self._inputs:
                if input["data"] not in self._copyin_cache:
                    self._copy_h2a(input["data"])

        if self._outputs:
            for output in self._outputs:
                if output["data"] not in self._alloc_cache:
                    self._alloc_h2a(output["data"])

        # run accel
        # TODO: ctypes for device and channel

        self._thread = threading.Thread(target=self._start_accel, args=(device, channel))
        self._thread.start()
        self._time_start = time.time()


    def wait(self, timeout=None):

        self._sharedlib.stop()

        if self._outputs:
            for output in self._outputs:
                if output not in self._copyout_cache:
                    self._copy_a2h(output)

        if timeout is None:
            thread.join()

        else:
            thread.join(max(0, timeout-time.time()+self._time_start))

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



