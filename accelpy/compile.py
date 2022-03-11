"""accelpy compile module"""

import os

from collections import OrderedDict
from accelpy.util import shellcmd, get_system

_compilers = OrderedDict()

_compilers["gnu_fortran_omptarget"] = OrderedDict()
_compilers["cray_fortran_omptarget"] = OrderedDict()
_compilers["amd_fortran_omptarget"] = OrderedDict()

def _gnu_version_check_omptarget(check):
    return check.lower().startswith(b"gnu")

def _cray_version_check_omptarget(check):
    return check.lower().startswith(b"cray")

def _amd_version_check_omptarget(check):
    return check.lower().startswith(b"amd")

system = get_system()

if system.name == "cray":

    _compilers["gnu_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version", _gnu_version_check_omptarget),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }

    _compilers["cray_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version",_cray_version_check_omptarget),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }

    _compilers["amd_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version",_cray_version_check_omptarget),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }


_compilers["gnu_fortran_omptarget"]["generic"] = {
        "check": ("gfortran --version", _gnu_version_check_omptarget),
        "build": "gfortran -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
    }

_compilers["cray_fortran_omptarget"]["generic"] = {
        "check": ("crayftn --version",_cray_version_check_omptarget),
        "build": "crayftn -shared -fPIC -h omp,noacc -J {moddir} -o {outpath}"
    }

_compilers["amd_fortran_omptarget"]["generic"] = {
        "check": ("amdflang --version",_cray_version_check_omptarget),
        "build": "amdflang -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
    }


def build_sharedlib(srcpath, outpath, compile=None, opts="", vendor=None, lang=None, accel=None):

    moddir = os.path.dirname(srcpath)

    if isinstance(compile, str):
        cmd = compile.format(moddir=moddir, outpath=outpath)
        return shellcmd(cmd + " " + srcpath)
        
    # user or system defined compilers

    for comptype, comps in _compilers.items():
        
        _vendor, _lang, _accel = comptype.split("_")

        if vendor is not None and vendor != _vendor: continue
        if lang is not None and lang != _lang: continue
        if accel is not None and accel != _accel: continue
        
        for compid, compinfo in comps.items():
            try:
                res = shellcmd(compinfo["check"][0])
                avail = compinfo["check"][1](res.stdout)
                if avail is None:
                    avail = compinfo["check"][1](res.stderr)

                if not avail: continue

                cmd = compinfo["build"].format(moddir=moddir, outpath=outpath)

                return shellcmd("%s %s %s" % (cmd, opts, srcpath))

            except Exception as err:
                print("command fail: %s" % cmd)

    raise Exception("All build commands were failed")

