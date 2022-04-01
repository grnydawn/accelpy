"""accelpy compile module"""

import os

from collections import OrderedDict
from tempfile import TemporaryDirectory
from accelpy.util import shellcmd, get_system

# NOTE: the priorities of the compilers are defined by the order of the compiler definitions

builtin_compilers = OrderedDict()

builtin_compilers["cray_fortran_omptarget"] = OrderedDict()
builtin_compilers["amd_fortran_omptarget"] = OrderedDict()
builtin_compilers["ibm_fortran_omptarget"] = OrderedDict()
builtin_compilers["gnu_fortran_omptarget"] = OrderedDict()

def _gnu_version_check_omptarget(check):
    return check.lower().startswith(b"gnu")

def _cray_version_check_omptarget(check):
    return check.lower().startswith(b"cray")

def _amd_version_check_omptarget(check):
    return check.lower().startswith(b"amd")

def _ibm_version_check_omptarget(check):
    return check.lower().startswith(b"ibm")

system = get_system()

if system.name == "cray":

    builtin_compilers["cray_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version",_cray_version_check_omptarget),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }

    builtin_compilers["amd_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version",_amd_version_check_omptarget),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }

    builtin_compilers["gnu_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version", _gnu_version_check_omptarget),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }

builtin_compilers["cray_fortran_omptarget"]["generic"] = {
        "check": ("crayftn --version",_cray_version_check_omptarget),
        "build": "crayftn -shared -fPIC -h omp,noacc -J {moddir} -o {outpath}"
    }

builtin_compilers["amd_fortran_omptarget"]["generic"] = {
        "check": ("amdflang --version",_amd_version_check_omptarget),
        "build": "amdflang -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
    }

builtin_compilers["ibm_fortran_omptarget"]["generic"] = {
        "check": ("xlf_r -qversion",_ibm_version_check_omptarget),
        "build": "xlf_r -qmkshrobj -qpic -qsmp=omp -qoffload -qmoddir={moddir} -o {outpath}"
    }

builtin_compilers["gnu_fortran_omptarget"]["generic"] = {
        "check": ("gfortran --version", _gnu_version_check_omptarget),
        "build": "gfortran -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
    }

def build_sharedlib(srcfile, outfile, workdir, compile=None, opts="", vendor=None, lang=None, accel=None):

    srcpath = os.path.join(workdir, srcfile)
    outpath = os.path.join(workdir, outfile)

    moddir = os.path.dirname(srcpath)

    if isinstance(compile, str):
        cmd = compile.format(moddir=moddir, outpath=outpath)
        out = shellcmd(cmd + " " + srcpath)

        if out.returncode == 0:
            return outpath

        else:
            return
        
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
        
        for compid, compinfo in comps.items():

            try:
                res = shellcmd(compinfo["check"][0])
                avail = compinfo["check"][1](res.stdout)
                if avail is None:
                    avail = compinfo["check"][1](res.stderr)

                #print(avail, compid, compinfo["check"])
                if not avail: continue

                cmd = compinfo["build"].format(moddir=moddir, outpath=outpath)

                out = shellcmd("%s %s %s" % (cmd, opts, srcpath), cwd=workdir)

                #import pdb; pdb.set_trace()
                if out.returncode == 0:
                    return outpath

                #print(str(out.stderr).replace("\\n", "\n"))
            except Exception as err:
                print("command fail: %s" % cmd)

    raise Exception("All build commands were failed")

