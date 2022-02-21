"""accelpy Spec module"""

import os, hashlib

from accelpy.util import Object, _accelpy_builtins, appeval, funcargseval, gethash


class Section():

    def __init__(self, accel, vargs, kwargs, body, env):
        self.accel = accel
        self.argnames = vargs
        self._kwargs = kwargs
        self.kwargs = None
        self.body = body
        self.env = env
        self.md5 = None

    def hash(self):

        if self.md5 is None:
            self.md5 = gethash("".join(self.body))

        return self.md5

    def is_enabled(self):
        return self.kwargs.get("enable", True)

    def kind(self):
        return self.accel 

    def update_argnames(self, data):

        for idx, arg in enumerate(data):

            if self.argnames is not None and len(self.argnames) > idx:
                arg["curname"] = self.argnames[idx]


class Spec(Object):

    def __init__(self, spec):

        # invargs, outvars, kwargs
        self._argnames = None

        if isinstance(spec, str):
            if os.path.isfile(spec):
                with open(spec) as fs:
                    spec = Spec(fs.read())

            self._sections = self._parse_spec(spec)

        elif isinstance(spec, Spec):
            self._sections = spec._sections 

        else:
            raise Exception("Wrong spec type: %s" % str(spec))

    def _of_set_argnames(self, *vargs):

        self._argnames = list(vargs)

    def _parse_spec(self, spec):

        rawlines = spec.split("\n")

        sec_starts = []

        for lineno, rawline in enumerate(rawlines):
            if rawline and rawline[0] == "[":
                    sec_starts.append(lineno)

        if len(sec_starts) == 0:
            raise Exception("No spec is found.")

        sec_starts.append(len(rawlines))

        self._env = dict(_accelpy_builtins)
        self._env["set_argnames"] =  self._of_set_argnames

        self.env = self._env.copy()

        self._pysection = rawlines[0:sec_starts[0]]

        sections = []
        for sec_start, sec_end in zip(sec_starts[0:-1], sec_starts[1:]):
            section = self._parse_section(rawlines[sec_start:sec_end])
            sections.extend(section)

        return sections

    def eval_pysection(self, environ):

        self.env = self._env.copy()

        if isinstance(environ, dict):
            self.env.update(environ)

        _, lenv = appeval("\n".join(self._pysection), self.env)

        self.env.update(lenv)

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
                    kwargs = [a.strip() for a in fargs[len(vargs):]]
                    body = rawlines[row+1:]
                    break

                except Exception as err:
                    row += 1
                    col = 0
        else:
            vargs, kwargs, body = [], {}, rawlines[row+1:]

        sections = []

        for accel in accels:
            sections.append(Section(accel, vargs, kwargs, body, self.env))

        return sections

    def update_argnames(self, data):

        for idx, arg in enumerate(data):

            if self._argnames is not None and len(self._argnames) > idx:
                arg["curname"] = self._argnames[idx]

            else:
                arg["curname"] = "accpy_var%d" % idx

    def get_section(self, accel):

        for sec in self._sections:

            sec.kwargs = funcargseval(",".join(sec._kwargs), self.env)

            if sec.accel == accel and sec.is_enabled():
                return sec

        raise Exception("No section with '%s' exists or is enabled." % accel)

    def list_sections(self, acctype=None):

        if isinstance(acctype, str):
            acctype = (acctype,)

        secs = []

        for sec in self._sections:

            sec.kwargs = funcargseval(",".join(sec._kwargs), self.env)

            if sec.is_enabled():
                if acctype and sec.accel not in acctype:
                    continue

                secs.append(sec.accel)

        return secs
