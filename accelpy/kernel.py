"""Accelpy kernel module"""

import os, sys, abc, tempfile, shutil

from collections import OrderedDict

from accelpy.util import Object, load_sharedlib, invoke_sharedlib, pack_arguments, shellcmd
from accelpy.compile import build_sharedlib, builtin_compilers

class KernelBase(Object):

    avails = OrderedDict()

    @abc.abstractmethod
    def gen_srcfiles(cls, section, workdir, copyinout, copyin, copyout, alloc):
        pass

class Task:

    def __init__(self, lang, libdata, copyinout, copyout):

        self.lang = lang
        self.libdata = libdata
        self.copyinout = copyinout
        self.copyout = copyout

    def stop(self):

        # invoke exit function in acceldata
        exitargs = []
        exitargs.extend([cio["data"] for cio in self.copyinout])
        exitargs.extend([co["data"] for co in self.copyout])

        resdata = invoke_sharedlib(self.lang, self.libdata, "dataexit", *exitargs)
        assert resdata == 0

class Kernel:

    def __init__(self, spec):

        self.spec = spec
        self._tasks = []
        self._workdir = tempfile.mkdtemp()


    def __del__(self):

        if os.path.isdir(self._workdir):
            shutil.rmtree(self._workdir)


    def stop(self):

        for task in self._tasks:
            task.stop()

    def launch(self, copyinout=None, copyin=None, copyout=None, alloc=None,
                compile=None, lang=None, vendor=None, accel=None, environ={}):

        # generate code and save to file

        self.spec.eval_pysection(environ)

        copyinout = pack_arguments(copyinout)
        copyin = pack_arguments(copyin)
        copyout = pack_arguments(copyout)
        alloc = pack_arguments(alloc)
            
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
            
            srcdata, srckernel = self._gen_srcfiles(_lang, _accel, copyinout,
                                    copyin, copyout, alloc)

            if srcdata is None or srckernel is None: continue

        #if isinstance(compile, str):
        #    return self._build_load_run(srcdata, srckernel, compile,
        #                                copyinout, copyin, copyout, alloc)

            for compid, compinfo in comps.items():
                try:
                    res = shellcmd(compinfo["check"][0])
                    avail = compinfo["check"][1](res.stdout)
                    if avail is None:
                        avail = compinfo["check"][1](res.stderr)

                    if not avail: continue

                    return self._build_load_run(_lang, srcdata, srckernel,
                                        compinfo["build"], copyinout, copyin,
                                        copyout, alloc)

                except Exception as err:
                    print("command fail: %s" % str(err))

        raise Exception("All build commands were failed")

    def _build_load_run(self, lang, srcdata, srckernel, command,
                            copyinout, copyin, copyout, alloc):

        libext = ".dylib" if sys.platform == "darwin" else ".so"

        # build acceldata
        dstdata = os.path.splitext(srcdata)[0] + libext
        cmd = command.format(moddir=self._workdir, outpath=dstdata)
        out = shellcmd(cmd + " " + srcdata, cwd=self._workdir)
        assert os.path.isfile(dstdata)

        # load acceldata
        libdata = load_sharedlib(dstdata)
        assert libdata is not None

        # invoke function in acceldata
        enterargs = []
        enterargs.extend([cio["data"] for cio in copyinout])
        enterargs.extend([ci["data"] for ci in copyin])
        enterargs.extend([co["data"] for co in copyout])
        enterargs.extend([a["data"] for a in alloc])

        resdata = invoke_sharedlib(lang, libdata, "dataenter", *enterargs)
        assert resdata == 0

        # build kernel
        dstkernel = os.path.splitext(srckernel)[0] + libext
        cmd = command.format(moddir=self._workdir, outpath=dstkernel)
        out = shellcmd(cmd + " " + srckernel, cwd=self._workdir)
        assert os.path.isfile(dstkernel)

        # load accelkernel
        libkernel = load_sharedlib(dstkernel)
        assert libkernel is not None

        # invoke function in accelkernel
        kernelargs = []
        kernelargs.extend([cio["data"] for cio in copyinout])
        kernelargs.extend([ci["data"] for ci in copyin])
        kernelargs.extend([co["data"] for co in copyout])
        kernelargs.extend([a["data"] for a in alloc])

        reskernel = invoke_sharedlib(lang, libkernel, "runkernel", *kernelargs)
        assert reskernel == 0

        task = Task(lang, libdata, copyinout, copyout)
        self._tasks.append(task)

    def _gen_srcfiles(self, lang, accel, copyinout, copyin, copyout, alloc):

        section = self.spec.get_section(accel, lang)

        if (section is None or section.lang not in KernelBase.avails or
            section.accel not in KernelBase.avails[section.lang]):
            return None, None

        self.spec.update_argnames(copyinout, copyin, copyout, alloc)
        section.update_argnames(copyinout, copyin, copyout, alloc)
 
        srcdata, srckernel = KernelBase.avails[section.lang][section.accel
                                ].gen_srcfiles(section, self._workdir,
                                copyinout, copyin, copyout, alloc)

        return srcdata, srckernel

