"""Accelpy kernel module"""

import os, sys, abc, tempfile, shutil, itertools

from collections import OrderedDict

from accelpy.const import version 
from accelpy.util import (Object, load_sharedlib, invoke_sharedlib,
                            pack_arguments, shellcmd, gethash, get_config) 
from accelpy.compile import build_sharedlib, builtin_compilers

class AccelBase(Object):

    avails = OrderedDict()

    @abc.abstractmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
                        copyin, copyout, alloc, attr):
        pass

    @abc.abstractmethod
    def gen_kernelfile(cls, datamodname, runid, section, workdir,
                        localvars, modvars):
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
                    accel=None, attr={}, recompile=False, _debug=False):

        self._id = next(self._ids)
        self._debug = _debug
        self._lang = None
        self._accel = None
        self._attr = attr
        self._tasks = {}
        self._workdir = tempfile.mkdtemp()
        self.debug("creating workdir: %s" % self._workdir)

        self.copyinout = pack_arguments(copyinout, prefix="cio%d_" % self._id)
        self.copyin = pack_arguments(copyin, prefix="ci%d_" % self._id)
        self.copyout = pack_arguments(copyout, prefix="co%d_" % self._id)
        self.alloc = pack_arguments(alloc, prefix="al%d_" % self._id)
            
        self._libkernel = None

        keys = [os.uname().release, version, self._id]

        for data in (self.copyinout, self.copyin, self.copyout, self.alloc):
            for item in data:
                keys.append(item["data"].shape)
                keys.append(item["data"].dtype)

        orghash = gethash(str(keys))

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

            libdir = get_config("libdir")

            self._dsrchash = gethash(orghash + str(comptype) + str(_lang) + str(_accel))
            
            dfname = "D" + self._dsrchash[2:] + AccelBase.avails[_lang][_accel].srcext
            self._dfdir = os.path.join(libdir, self._dsrchash[:2])
            dsrcpath = os.path.join(self._dfdir, dfname)
            self._dmodname = "mod" + self._dsrchash[2:]

            if os.path.isfile(dsrcpath) and not recompile: 
                srcdata = dsrcpath

            else:
                srcdata = AccelBase.avails[_lang][_accel].gen_datafile(
                        self._dmodname, dfname, self._id, self._workdir,
                        self.copyinout, self.copyin, self.copyout, self.alloc,
                        self._attr)

            if srcdata is None: continue

            self.debug("data src from: %s" % srcdata)

            for compid, compinfo in comps.items():
                try:
                    res = shellcmd(compinfo["check"][0])
                    avail = compinfo["check"][1](res.stdout)
                    if avail is None:
                        avail = compinfo["check"][1](res.stderr)

                    if not avail: continue

                    cmdline = compinfo["build"]

                    self._dlibhash = gethash(self._dsrchash +str(compid) + str(cmdline))
                    dlname = "D" + self._dlibhash[2:] + AccelBase.avails[_lang][_accel].libext
                    dldir = os.path.join(libdir, self._dlibhash[:2])
                    dlibpath = os.path.join(dldir, dlname)

                    if os.path.isfile(dlibpath) and not recompile: 
                        self._libdata = self._load_run_enter(dlibpath, _lang)
                        self.debug("data lib from lib: %s" % dlibpath)

                    else:
                        dstpath = os.path.join(self._workdir, dlname)
                        modname, self._libdata = self._build_run_enter(dstpath, _lang,
                                            srcdata, cmdline)
                        self.debug("data lib from build: %s" % dstpath)

                        if not os.path.isfile(dsrcpath):
                            if not os.path.isdir(self._dfdir):
                                os.makedirs(self._dfdir)

                            shutil.copy(srcdata, dsrcpath)

                            if modname:
                                shutil.copy(os.path.join(self._workdir,
                                        modname), self._dfdir)

                        if not os.path.isfile(dlibpath):
                            if not os.path.isdir(dldir):
                                os.makedirs(dldir)

                            shutil.copy(dstpath, dlibpath)

                    self._compile = cmdline
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
        #exitargs.extend([cio["data"] for cio in self.copyinout])
        #exitargs.extend([co["data"] for co in self.copyout])

        self.debug("libdata exit sharedlib", self._libdata)
        resdata = invoke_sharedlib(self._lang, self._libdata, "dataexit_%d" % self._id, *exitargs)
        #del self.libdata

        self.debug("after dataexit cio", *[cio["data"] for cio in self.copyinout])
        self.debug("after dataexit co", *[co["data"] for co in self.copyout])

        assert resdata == 0, "dataexit invoke fail"

        #for task in self._tasks:
        #    task.stop()

    def launch(self, spec, *kargs, macro={}, environ={}):

        localvars = pack_arguments(kargs)

        self.spec = spec
        self.spec.eval_pysection(environ)
        self.spec.update_argnames(localvars)
        self.section = self.spec.get_section(self._accel, self._lang, environ)

        if (self.section is None or self._lang not in AccelBase.avails or
            self._accel not in AccelBase.avails[self._lang]):
            raise Exception("Kernel can not be crated.")

        self.section.update_argnames(localvars)

        dids = {}
        _kargs = []
        _uonly = []

        dids = dict((d["id"], d["curname"]) for d in
                        self.copyinout+self.copyin+self.copyout+self.alloc)

        for lvar in localvars:

            if lvar["id"] in dids:
                _uonly.append((dids[lvar["id"]], lvar))

            else:
                _kargs.append(lvar)


        keys = [os.uname().release, version, self._dlibhash]
        #keys.extend([u[1] for u in _uonly])
        keys.extend(_uonly)

        for item in _kargs:
            keys.append(item["data"].shape)
            keys.append(item["data"].dtype)

        self._knlhash = gethash(self.section.hash() + str(keys) + str(macro) +
                            str(environ))

        libdir = get_config("libdir")

        klib = None

        klname = "K" + self._knlhash[2:] + AccelBase.avails[
                        self._lang][self._accel].libext
        kldir = os.path.join(libdir, self._knlhash[:2])
        klibpath = os.path.join(kldir, klname)

        if self._knlhash in self._tasks:
            self._run_kernel(self._tasks[self._knlhash], _kargs)
            self.debug("kernel run using memory cache")

        elif os.path.isfile(klibpath): 
            klib = self._load_run_kernel(klibpath, _kargs)
            self._tasks[self._knlhash] = klib
            self.debug("kernel run using %s" % klibpath)

        else:

            self.macro = macro

            srckernel = AccelBase.avails[self._lang][self._accel
                                ].gen_kernelfile(self._knlhash, self._dmodname,
                                self._id, self.section, self._workdir, _kargs,
                                _uonly)

            dstpath, klib = self._build_load_run_kernel(srckernel, _kargs)

            if not os.path.isfile(klibpath):
                if not os.path.isdir(kldir):
                    os.makedirs(kldir)

                shutil.copy(dstpath, klibpath)
            self.debug("kernel run using %s" % dstpath)

        if klib and self._knlhash not in self._tasks:
            self._tasks[self._knlhash] = klib

    def _load_run_enter(self, dstdata, lang):

        # load acceldata
        libdata = load_sharedlib(dstdata)
        self.debug("libdata sharedlib", libdata)
        assert libdata is not None, "libdata load fail"

        # invoke function in acceldata
        enterargs = []
        enterargs.extend([cio["data"] for cio in self.copyinout])
        enterargs.extend([ci["data"] for ci in self.copyin])
        enterargs.extend([co["data"] for co in self.copyout])
        enterargs.extend([al["data"] for al in self.alloc])

        resdata = invoke_sharedlib(lang, libdata, "dataenter_%d" % self._id, *enterargs)
        assert resdata == 0, "dataenter invoke fail"

        #task = Task(lang, libdata, self.copyinout, self.copyout, self._debug)
        #self._tasks.append(task)

        return libdata

    def _build_run_enter(self, dstdata, lang, srcdata, command):

        # build acceldata
        cmd = command.format(moddir=self._workdir, outpath=dstdata)
        out = shellcmd(cmd + " " + srcdata, cwd=self._workdir)
        assert os.path.isfile(dstdata), str(out.stderr)

#        shutil.copy(dstdata, os.path.join(self._workdir, "libdata.so"))
#        dstdataobj = os.path.join(os.path.dirname(srcdata), "data.o")
#        cmd2 = command.format(moddir=self._workdir, outpath=dstdataobj)
#        out2 = shellcmd(cmd2 + " -c " + srcdata, cwd=self._workdir)
#        assert os.path.isfile(dstdataobj), str(out2.stderr)

        modname = None

        for fname in os.listdir(self._workdir):
            if fname.lower().endswith(".mod"):
                modname = fname
                break

        return modname, self._load_run_enter(dstdata, lang)

    def _build_load_run_kernel(self, srckernel, localvars):

        libext = AccelBase.avails[self._lang][self._accel].libext

        macros = []
        for m, d in self.macro.items():
            if d is None or d is False:
                continue

            if d is True:
                d = 1

            macros.append("-D%s=%s" % (str(m), str(d)))
        
        include = "-I %s" % self._dfdir

        # build kernel
        dstkernel = os.path.splitext(srckernel)[0] + libext
        cmd = self._compile.format(moddir=self._workdir, outpath=dstkernel)
        out = shellcmd("%s %s %s %s" % (cmd, " ".join(macros), include,
                        srckernel), cwd=self._workdir)
        #print(str(out.stdout).replace("\\n", "\n"))
        #print(str(out.stderr).replace("\\n", "\n"))
        #import pdb; pdb.set_trace()
        assert os.path.isfile(dstkernel), str(out.stderr).replace("\\n", "\n")

        return dstkernel, self._load_run_kernel(dstkernel, localvars)

    def _load_run_kernel(self, libpath, localvars):

        # load accelkernel
        libkernel = load_sharedlib(libpath)
        self.debug("libkernel sharedlib", libkernel)
        assert libkernel is not None, "libkernel load fail"

        self._run_kernel(libkernel, localvars)

        return libkernel


    def _run_kernel(self, libkernel, localvars):

        # invoke function in accelkernel
        kernelargs = [lvar["data"] for lvar in localvars]

        self.debug("before kernel", *kernelargs)

        reskernel = invoke_sharedlib(self._lang, libkernel, "runkernel_%d" % self._id, *kernelargs)

        self.debug("after kernel cio", *kernelargs)

        assert reskernel == 0, "runkernel invoke fail"

