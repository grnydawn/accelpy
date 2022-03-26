"""Accelpy kernel module"""

import os, sys, abc, tempfile, shutil, itertools

from collections import OrderedDict

from accelpy.util import Object, load_sharedlib, invoke_sharedlib, pack_arguments, shellcmd
from accelpy.compile import build_sharedlib, builtin_compilers

class AccelBase(Object):

    avails = OrderedDict()

    @abc.abstractmethod
    def gen_datafile(cls, workdir, copyinout, copyin, copyout, alloc):
        pass

    @abc.abstractmethod
    def gen_kernelfile(cls, section, workdir, localvars):
        pass

class Task:

    def __init__(self, lang, libdata, copyinout, copyout, _debug):

        self._debug = _debug
        self.lang = lang
        self.libdata = libdata
        self.copyinout = copyinout
        self.copyout = copyout

    def debug(self, *objs):

        if self._debug:
            print("DEBUG: " + " ".join([str(o) for o in objs]))
#
#    def stop(self):
#
#        return self.wait()
#
#    def wait(self):
#
#        # invoke exit function in acceldata
#        exitargs = []
#        exitargs.extend([cio["data"] for cio in self.copyinout])
#        exitargs.extend([co["data"] for co in self.copyout])
#
#        resdata = invoke_sharedlib(self.lang, self.libdata, "dataexit", *exitargs)
#        #del self.libdata
#
#        self.debug("after dataexit cio", *[cio["data"] for cio in self.copyinout])
#        self.debug("after dataexit co", *[co["data"] for co in self.copyout])
#
#        assert resdata == 0, "dataexit invoke fail"


class Accel:

    _ids = itertools.count(0)

    def __init__(self, copyinout=None, copyin=None, copyout=None,
                    alloc=None, compile=None, lang=None, vendor=None,
                    accel=None, _debug=False):

        self._id = next(self._ids)
        self._debug = _debug
        self._lang = None
        self._accel = None
        #self._tasks = []
        self._tasks = {}
        self._workdir = tempfile.mkdtemp()
        self.debug("creating workdir: %s" % self._workdir)

        self.copyinout = pack_arguments(copyinout, prefix="cio%d" % self._id)
        self.copyin = pack_arguments(copyin, prefix="ci%d" % self._id)
        self.copyout = pack_arguments(copyout, prefix="co%d" % self._id)
        self.alloc = pack_arguments(alloc, prefix="al%d" % self._id)
            
        self._libkernel = None

        # user or system defined compilers
        for comptype, comps in builtin_compilers.items():
            
            _vendor, _lang, _accel = comptype.split("_")

            if isinstance(vendor, (list, tuple)):
                if _vendor not in vendor: continue

            elif isinstance(vendor, str):
                if _vendor != vendor: continue

            if isinstance(lang, (list, tuple)):
                if _lang not in lang: continue

            elif isinstance(lang, str):
                if _lang != lang: continue

            if isinstance(accel, (list, tuple)):
                if _accel not in accel: continue

            elif isinstance(accel, str):
                if _accel != accel: continue
            
            srcdata = AccelBase.avails[_lang][_accel].gen_datafile(self._workdir,
                        self.copyinout, self.copyin, self.copyout, self.alloc)

            if srcdata is None: continue

        #if isinstance(compile, str):
        #    return self._build_load_run(srcdata, srckernel, compile,
        #                                copyinout, copyin, copyout, alloc)

            for compid, compinfo in comps.items():
                try:
                    res = shellcmd(compinfo["check"][0])
                    avail = compinfo["check"][1](res.stdout)
                    if avail is None:
                        avail = compinfo["check"][1](res.stderr)

                    #print(avail, compid, compinfo["check"])
                    if not avail: continue

                    self._libdata = self._build_run_enter(_lang, srcdata,
                                        compinfo["build"])

                    self._compile = compinfo["build"]
                    self._lang = _lang
                    self._accel = _accel

                    return 

                except Exception as err:
                    print("INFO: unsuccessful compiler command: %s" % str(err))

        raise Exception("All build commands for enterdata were failed")

    def __del__(self):

        if os.path.isdir(self._workdir):
            self.debug("removing workdir: %s" % self._workdir)
            shutil.rmtree(self._workdir)

    def debug(self, *objs):

        if self._debug:
            print("DEBUG: " + " ".join([str(o) for o in objs]))

    def stop(self):

        # invoke exit function in acceldata
        exitargs = []
        exitargs.extend([cio["data"] for cio in self.copyinout])
        exitargs.extend([co["data"] for co in self.copyout])

        resdata = invoke_sharedlib(self._lang, self._libdata, "dataexit", *exitargs)
        #del self.libdata

        self.debug("after dataexit cio", *[cio["data"] for cio in self.copyinout])
        self.debug("after dataexit co", *[co["data"] for co in self.copyout])

        assert resdata == 0, "dataexit invoke fail"

        #for task in self._tasks:
        #    task.stop()

    def launch(self, spec, *kargs, macro={}, environ={}):

        localvars = pack_arguments(kargs)

        if spec in self._tasks:
            _libkernel = self._tasks[spec]

        else:

            self.spec = spec
            self.spec.eval_pysection(environ)
            self.spec.update_argnames(localvars)
            self.section = self.spec.get_section(self._accel, self._lang)
            self.section.update_argnames(localvars)

            if (self.section is None or self._lang not in AccelBase.avails or
                self._accel not in AccelBase.avails[self._lang]):
                return

            self.macro = macro

            srckernel = AccelBase.avails[self._lang][self._accel
                                    ].gen_kernelfile(self.section, self._workdir,
                                    localvars)

            _libkernel = self._build_kernel(srckernel)
            self._tasks[spec] = _libkernel

        self._run_kernel(localvars, _libkernel)

    def _build_run_enter(self, lang, srcdata, command):

        libext = ".dylib" if sys.platform == "darwin" else ".so"

        # build acceldata
        dstdata = os.path.splitext(srcdata)[0] + libext
        cmd = command.format(moddir=self._workdir, outpath=dstdata)
        out = shellcmd(cmd + " " + srcdata, cwd=self._workdir)
        #import pdb; pdb.set_trace()
        assert os.path.isfile(dstdata), str(out.stderr)

        # load acceldata
        libdata = load_sharedlib(dstdata)
        self.debug("libdata sharedlib", libdata)
        assert libdata is not None, "libdata load fail"

        # invoke function in acceldata
        enterargs = []
        enterargs.extend([cio["data"] for cio in self.copyinout])
        enterargs.extend([ci["data"] for ci in self.copyin])
        enterargs.extend([co["data"] for co in self.copyout])
        enterargs.extend([a["data"] for a in self.alloc])

        resdata = invoke_sharedlib(lang, libdata, "dataenter", *enterargs)
        assert resdata == 0, "dataenter invoke fail"

        #task = Task(lang, libdata, self.copyinout, self.copyout, self._debug)
        #self._tasks.append(task)

        return libdata

    def _build_kernel(self, srckernel):

        libext = ".dylib" if sys.platform == "darwin" else ".so"

        macros = []
        for m, d in self.macro.items():
            if d is None or d is False:
                continue

            if d is True:
                d = 1

            macros.append("-D%s=%s" % (str(m), str(d)))
        
        # build kernel
        dstkernel = os.path.splitext(srckernel)[0] + libext
        cmd = self._compile.format(moddir=self._workdir, outpath=dstkernel)
        out = shellcmd("%s %s %s" % (cmd, " ".join(macros), srckernel),
                        cwd=self._workdir)
        #print(str(out.stdout).replace("\\n", "\n"))
        #import pdb; pdb.set_trace()
        #assert os.path.isfile(dstkernel), str(out.stderr)
        assert os.path.isfile(dstkernel), str(out.stderr).replace("\\n", "\n")

        # load accelkernel
        libkernel = load_sharedlib(dstkernel)
        self.debug("libkernel sharedlib", libkernel)
        assert libkernel is not None, "libkernel load fail"

        return libkernel


    def _run_kernel(self, localvars, libkernel):

        # invoke function in accelkernel
        kernelargs = [lvar["data"] for lvar in localvars]

        self.debug("before kernel", *kernelargs)

        reskernel = invoke_sharedlib(self._lang, libkernel, "runkernel", *kernelargs)

        self.debug("after kernel cio", *kernelargs)

        assert reskernel == 0, "runkernel invoke fail"

