"""accelpy Kernel module"""

import os, hashlib

from accelpy.util import Object, _accelpy_builtins, appeval, funcargseval, gethash


class Section():

    def __init__(self, accel, lang, fargs, body):

        self.accel = accel
        self.lang = lang
        self.fargs = fargs
        self.body = body
        self.md5 = None

    def hash(self):

        if self.md5 is None:
            self.md5 = gethash("".join(self.body))

        return self.md5

    def is_enabled(self):
        return self.kwargs.get("enable", True)

    def kind(self):
        return self.accel 

    def update_argnames(self, args):

        for idx in range(len(self.vargs)):
            args[idx]["curname"] = self.vargs[idx]


class Kernel(Object):

    def __init__(self, spec):

        # invargs, outvars, kwargs
        self._argnames = []

        if isinstance(spec, str):
            if os.path.isfile(spec):
                with open(spec) as fs:
                    spec = Kernel(fs.read())

            self._sections = self._parse_spec(spec)

        elif isinstance(spec, Kernel):
            self._sections = spec._sections 

        else:
            raise Exception("Wrong spec type: %s" % str(spec))

    def _of_set_argnames(self, *vargs, **kwargs):

        self._argnames.extend(vargs)

    def _parse_spec(self, spec):

        rawlines = spec.split("\n")

        sec_starts = []

        for lineno, rawline in enumerate(rawlines):
            if rawline and rawline[0] == "[":
                    sec_starts.append(lineno)

        if len(sec_starts) == 0:
            raise Exception("No spec is found.")

        sec_starts.append(len(rawlines))

        self._pysection = rawlines[0:sec_starts[0]]

        sections = []
        for sec_start, sec_end in zip(sec_starts[0:-1], sec_starts[1:]):
            section = self._parse_section(rawlines[sec_start:sec_end])
            sections.extend(section)

        return sections

    def eval_pysection(self, env):

        self.env = dict(_accelpy_builtins)
        self.env["set_argnames"] =  self._of_set_argnames

        if isinstance(env, dict):
            self.env.update(env)

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

        #accels = [n.strip() for n in "".join(names).split(",")]
        accels = []
        for accid in "".join(names).split(","):
            acclang = accid.strip().split("_")
            if len(acclang) == 2:
                accels.append(acclang)
    
            elif len(acclang) == 1:
                accels.append(acclang*2)

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
                    body = rawlines[row+1:]
                    break

                except Exception as err:
                    row += 1
                    col = 0
        else:
            fargs, body = [], rawlines[row+1:]

        sections = []

        for acc, lang in accels:
            sections.append(Section(acc, lang, fargs, body))

        return sections

    def update_argnames(self, args):

        for idx in range(len(self._argnames)):
            args[idx]["curname"] = self._argnames[idx]

    def get_section(self, accel, lang):

        for sec in self._sections:

            if sec.accel != accel or sec.lang != lang:
                continue

            if not hasattr(self, "vargs") or not hasattr(self, "kwargs"):
                sec.vargs, sec.kwargs = funcargseval(",".join(sec.fargs), self.env)

            if sec.is_enabled():
                return sec

