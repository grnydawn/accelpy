"""Accelpy kernel module"""

import os, sys, abc, tempfile, shutil, itertools

from collections import OrderedDict

from accelpy.const import version 
from accelpy.util import (Object, load_sharedlib, invoke_sharedlib,
                            pack_arguments, shellcmd, gethash, get_config) 
from accelpy.compile import build_sharedlib, builtin_compilers

class AccelBase(Object):

    libext = ".dylib" if sys.platform == "darwin" else ".so"
    avails = OrderedDict()

    @abc.abstractmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
                        copyin, copyout, alloc, attr):
        pass

    @abc.abstractmethod
    def gen_kernelfile(cls, datamodname, runid, section, workdir,
                        localvars, modvars):
        pass


class Accel:

    _ids = itertools.count(0)

    def __init__(self, copyinout=None, copyin=None, copyout=None,
                    alloc=None, compile=None, lang=[], vendor=[],
                    accel=[], attr={}, recompile=False, _debug=False):

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

        if not isinstance(vendor, (list, tuple)):
            vendor = (vendor, )

        if not isinstance(lang, (list, tuple)):
            lang = (lang, )

        if not isinstance(accel, (list, tuple)):
            accel = (accel, )

        # user or system defined compilers
        for comptype, comps in builtin_compilers.items():
            
            _vendor, _lang, _accel = comptype.split("_")

            #print(vendor, _vendor, lang, _lang, accel, _accel)

            if (vendor and "*" not in vendor and _vendor != "*" and
                _vendor not in vendor): continue

            if (lang and "*" not in lang and _lang != "*" and
                _lang not in lang): continue

            if (accel and "*" not in accel and _accel != "*" and
                _accel not in accel): continue

            libdir = get_config("libdir")

            self._dsrchash = gethash(orghash + str(comptype) + str(_lang) + str(_accel))
            
            dfname = "D" + self._dsrchash[2:] + AccelBase.avails[_lang][_accel].srcext
            self._dfdir = os.path.join(libdir, self._dsrchash[:2])
            dsrcpath = os.path.join(self._dfdir, dfname)
            self._dmodname = "mod" + self._dsrchash[2:]

            #import pdb; pdb.set_trace()

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

                    print("Compiling with %s %s compiler for %s target."
                            % (_vendor, _lang, _accel))
    
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

        if hasattr(self, "_workdir") and os.path.isdir(self._workdir):
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

        self.debug("libdata exit sharedlib", self._libdata)
        resdata = invoke_sharedlib(self._lang, self._libdata, "dataexit_%d" % self._id, *exitargs)

        self.debug("after dataexit cio", *[cio["data"] for cio in self.copyinout])
        self.debug("after dataexit co", *[co["data"] for co in self.copyout])

        assert resdata == 0, "dataexit invoke fail"

    def launch(self, spec, *kargs, updateto=None, updatefrom=None, macro={}, environ={}):

        localvars = pack_arguments(kargs, updateto=updateto, updatefrom=updatefrom)

        self.spec = spec
        self.spec.eval_pysection(environ)
        self.spec.update_argnames(localvars)
        self.section = self.spec.get_section(self._accel, self._lang, environ)

        if (self.section is None or self._lang not in AccelBase.avails or
            self._accel not in AccelBase.avails[self._lang]):
            raise Exception("Kernel can not be created.")

        self.section.update_argnames(localvars)

        keys = [os.uname().release, version, self._dlibhash]

        # TODO: key gen from environ and Accel env

        dids = {}
        _kargs = []
        _uonly = []
        _mvars = []

        dids = dict((d["id"], d["curname"]) for d in
                        self.copyinout+self.copyin+self.copyout+self.alloc)

        for lvar in localvars:

            if lvar["id"] in dids:
                _mvars.append(lvar)
                _uonly.append((dids[lvar["id"]], lvar))
                keys.append((dids[lvar["id"]], lvar["curname"]))

            else:
                _kargs.append(lvar)
                keys.append(lvar["data"].shape)
                keys.append(lvar["data"].dtype)

            keys.append(lvar["updateto"])
            keys.append(lvar["updatefrom"])

        self._knlhash = gethash(self.section.hash() + str(keys) + str(macro) +
                            str(environ))

        libdir = get_config("libdir")

        klib = None

        klname = "K" + self._knlhash[2:] + AccelBase.avails[
                        self._lang][self._accel].libext
        kldir = os.path.join(libdir, self._knlhash[:2])
        klibpath = os.path.join(kldir, klname)

        if self._knlhash in self._tasks:
            self._run_kernel(self._tasks[self._knlhash], _kargs, _mvars)
            self.debug("kernel run using memory cache")

        elif os.path.isfile(klibpath): 
            klib = self._load_run_kernel(klibpath, _kargs, _mvars)
            self._tasks[self._knlhash] = klib
            self.debug("kernel run using %s" % klibpath)

        else:

            self.macro = macro
            modattr = dict(self._attr)

            srckernel = AccelBase.avails[self._lang][self._accel
                                ].gen_kernelfile(self._knlhash, self._dmodname,
                                self._id, self.spec._id, modattr, self.section, self._workdir, _kargs,
                                _uonly)

            dstpath, klib = self._build_load_run_kernel(srckernel, _kargs, _mvars)

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

        #import pdb;pdb.set_trace()
        resdata = invoke_sharedlib(lang, libdata, "dataenter_%d" % self._id, *enterargs)
        assert resdata == 0, "dataenter invoke fail"

        return libdata

    def _build_run_enter(self, dstdata, lang, srcdata, command):

        # build acceldata
        cmd = command.format(moddir=self._workdir, outpath=dstdata)
        out = shellcmd(cmd + " " + srcdata, cwd=self._workdir)
        #print(str(out.stdout).replace("\\n", "\n"))
        #print(str(out.stderr).replace("\\n", "\n"))
        #import pdb; pdb.set_trace()
        assert os.path.isfile(dstdata), str(out.stderr)

        modname = None

        for fname in os.listdir(self._workdir):
            if fname.lower().endswith(".mod"):
                modname = fname
                break

        return modname, self._load_run_enter(dstdata, lang)

    def _build_load_run_kernel(self, srckernel, localvars, modvars):

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

        return dstkernel, self._load_run_kernel(dstkernel, localvars, modvars)

    def _load_run_kernel(self, libpath, localvars, modvars):

        # load accelkernel
        libkernel = load_sharedlib(libpath)
        self.debug("libkernel sharedlib", libkernel)
        assert libkernel is not None, "libkernel load fail"

        self._run_kernel(libkernel, localvars, modvars)

        return libkernel


    def _run_kernel(self, libkernel, localvars, modvars):

        # invoke function in accelkernel
        kernelargs = [mvar["data"] for mvar in modvars]
        kernelargs.extend([lvar["data"] for lvar in localvars])

        self.debug("before kernel", *kernelargs)
        #import pdb; pdb.set_trace()
        reskernel = invoke_sharedlib(self._lang, libkernel,
                        "runkernel_%d%d" % (self._id, self.spec._id),  *kernelargs)

        self.debug("after kernel cio", *kernelargs)

        assert reskernel == 0, "runkernel invoke fail"

