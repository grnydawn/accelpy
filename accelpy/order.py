"""accelpy Order module"""

import os

from accelpy.core import Object
from accelpy.util import _accelpy_builtins, appeval, funcargseval


class Section():

    def __init__(self, accel, vargs, kwargs, body):
        self.accel = accel
        self.argnames = vargs
        self.kwargs = kwargs
        self.body = body

    def is_enabled(self):
        return kwargs.get("enable", True)

    def kind(self):
        return self.accel 

    def update_argnames(self, inputs, outputs):

        for idx, data in enumerate(inputs+outputs):

            if self.argnames is not None and len(self.argnames) > idx:
                data["curname"] = self.argnames[idx]


class Order(Object):

    def __init__(self, order, env_order= {}):

        # invargs, outvars, kwargs
        self._argnames = None

        if isinstance(order, str):
            if os.path.isfile(order):
                with open(order) as fo:
                    order = Order(fo.read())

            self._sections = self._parse_order(order, env_order)

        elif isinstance(order, Order):
            self._sections = order._sections 

        else:
            raise Exception("Wrong order type: %s" % str(order))

    def _of_set_argnames(self, *vargs):

        self._argnames = list(vargs)

    def _parse_order(self, order, env_order):

        rawlines = order.split("\n")

        sec_starts = []

        for lineno, rawline in enumerate(rawlines):
            if rawline and rawline[0] == "[":
                    sec_starts.append(lineno)

        if len(sec_starts) == 0:
            raise Exception("No order is found.")

        sec_starts.append(len(rawlines))

        self._env = dict(_accelpy_builtins)
        self._env["set_argnames"] =  self._of_set_argnames
        if isinstance(env_order, dict):
            self._env.update(env_order)

        secpy = rawlines[0:sec_starts[0]]
        _, lenv = appeval("\n".join(secpy), self._env)

        self._env.update(lenv)

        sections = []
        for sec_start, sec_end in zip(sec_starts[0:-1], sec_starts[1:]):
            section = self._parse_section(rawlines[sec_start:sec_end])
            sections.extend(section)

        return sections

    def _parse_section(self, rawlines):

        assert (rawlines[0] and rawlines[0][0] == "[")

        maxlineno = len(rawlines)
        row = 0
        col = 1

        names = []
        arg_start = None

        # collect accelerator names
        while(row < maxlineno):

            if rawlines[row].lstrip().startswith("#"):
                row += 1
                col = 0
                continue

            for idx in range(col, len(rawlines[row])):
                c = rawlines[row][idx]
                if c in (":", "]"):
                    names.append(rawlines[row][col:idx])
                    arg_start = [c, row, idx+1]
                    row = maxlineno
                    break
            if row < maxlineno:
                names.append(rawlines[row])
            row += 1
            col = 0

        assert names

        accels = [n.strip() for n in "".join(names).split(",")]
        
        args = []

        char, row, col = arg_start

        # collect accelerator arguments
        if char != "]":

            while(row < maxlineno):

                line = rawlines[row][col:].rstrip()

                if not line or line[-1] != "]":
                    args.append(line)
                    row += 1
                    col = 0
                    continue
                else:
                    args.append(line[:-1])

                try:
                    fargs = " ".join(args).split(",")
                    vargs = [a.strip() for a in fargs if "=" not in a]
                    _kwargs = [a.strip() for a in fargs[len(vargs):]]
                    _, kwargs = funcargseval(",".join(_kwargs), self._env)
                    body = rawlines[row+1:]
                    break

                except Exception as err:
                    row += 1
                    col = 0
        else:
            vargs, kwargs, body = [], {}, rawlines[row+1:]

        sections = []

        for accel in accels:
            sections.append(Section(accel, vargs, kwargs, body))

        return sections

    def update_argnames(self, inputs, outputs):

        for idx, data in enumerate(inputs+outputs):

            if self._argnames is not None and len(self._argnames) > idx:
                data["curname"] = self._argnames[idx]

            else:
                data["curname"] = "accpy_var%d" % idx

    def get_section(self, accel):

        for sec in self._sections:
            if sec.accel == accel and getattr(sec, "enable", True):
                return sec

        raise Exception("No section with '%s' exists or is enabled." % accel)
