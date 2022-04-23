"""accelpy compile module"""

import os, sys

from collections import OrderedDict
from tempfile import TemporaryDirectory
from accelpy.util import shellcmd, get_system

# NOTE: the priorities of the compilers are defined by the order of the compiler definitions

builtin_compilers = OrderedDict()

builtin_compilers["*_cpp_cuda"] = OrderedDict()
builtin_compilers["*_cpp_hip"] = OrderedDict()

builtin_compilers["cray_fortran_omptarget"] = OrderedDict()
builtin_compilers["cray_cpp_omptarget"] = OrderedDict()
builtin_compilers["cray_fortran_openacc"] = OrderedDict()
builtin_compilers["cray_cpp_openacc"] = OrderedDict()
builtin_compilers["cray_fortran_openmp"] = OrderedDict()
builtin_compilers["cray_cpp_openmp"] = OrderedDict()
builtin_compilers["cray_fortran_fortran"] = OrderedDict()
builtin_compilers["cray_cpp_cpp"] = OrderedDict()

builtin_compilers["amd_fortran_omptarget"] = OrderedDict()
builtin_compilers["amd_cpp_omptarget"] = OrderedDict()
builtin_compilers["amd_fortran_openacc"] = OrderedDict()
builtin_compilers["amd_cpp_openacc"] = OrderedDict()
builtin_compilers["amd_fortran_openmp"] = OrderedDict()
builtin_compilers["amd_cpp_openmp"] = OrderedDict()
builtin_compilers["amd_fortran_fortran"] = OrderedDict()
builtin_compilers["amd_cpp_cpp"] = OrderedDict()

builtin_compilers["ibm_fortran_omptarget"] = OrderedDict()
builtin_compilers["ibm_cpp_omptarget"] = OrderedDict()
builtin_compilers["ibm_fortran_openmp"] = OrderedDict()
builtin_compilers["ibm_cpp_openmp"] = OrderedDict()
builtin_compilers["ibm_fortran_fortran"] = OrderedDict()
builtin_compilers["ibm_cpp_cpp"] = OrderedDict()

builtin_compilers["intel_fortran_omptarget"] = OrderedDict()
builtin_compilers["intel_cpp_omptarget"] = OrderedDict()
builtin_compilers["intel_fortran_openmp"] = OrderedDict()
builtin_compilers["intel_cpp_openmp"] = OrderedDict()
builtin_compilers["intel_fortran_fortran"] = OrderedDict()
builtin_compilers["intel_cpp_cpp"] = OrderedDict()

builtin_compilers["pgi_fortran_omptarget"] = OrderedDict()
builtin_compilers["pgi_cpp_omptarget"] = OrderedDict()
builtin_compilers["pgi_fortran_openacc"] = OrderedDict()
builtin_compilers["pgi_cpp_openacc"] = OrderedDict()
builtin_compilers["pgi_fortran_openmp"] = OrderedDict()
builtin_compilers["pgi_cpp_openmp"] = OrderedDict()
builtin_compilers["pgi_fortran_fortran"] = OrderedDict()
builtin_compilers["pgi_cpp_cpp"] = OrderedDict()

builtin_compilers["gnu_fortran_omptarget"] = OrderedDict()
builtin_compilers["gnu_cpp_omptarget"] = OrderedDict()
builtin_compilers["gnu_fortran_openacc"] = OrderedDict()
builtin_compilers["gnu_cpp_openacc"] = OrderedDict()
builtin_compilers["gnu_fortran_openmp"] = OrderedDict()
builtin_compilers["gnu_cpp_openmp"] = OrderedDict()
builtin_compilers["gnu_fortran_fortran"] = OrderedDict()
builtin_compilers["gnu_cpp_cpp"] = OrderedDict()



def _gnu_version_check(check):

    if sys.platform == "darwin":
        return (check.lower().lstrip().startswith(b"gnu") or
            check.lower().lstrip().startswith(b"configured"))

    else:
        return (check.lower().lstrip().startswith(b"gnu") or
            check.lower().lstrip().startswith(b"g++"))


def _pgi_version_check(check):
    return check.lower().lstrip().startswith(b"pg")

def _cuda_check(check):
    return check.lower().lstrip().startswith(b"nvcc")

def _cray_version_check(check):
    return check.lower().lstrip().startswith(b"cray")

def _amd_version_check(check):
    return check.lower().lstrip().startswith(b"amd")

def _hip_check(check):
    return check.lower().lstrip().startswith(b"hip")

def _ibm_version_check(check):
    return check.lower().lstrip().startswith(b"ibm")

def _intel_version_check(check):
    temp = check.lower().lstrip()
    return temp.startswith(b"ifort") or temp.startswith(b"icpc")

system = get_system()

if system.name == "cray":

    builtin_compilers["cray_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version",_cray_version_check),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }

    builtin_compilers["cray_cpp_omptarget"]["CCwrapper"] = {
            "check": ("CC --version",_cray_version_check),
            "build": "CC -shared -fPIC -fopenmp -o {outpath}"
        }

    builtin_compilers["cray_fortran_openacc"]["ftnwrapper"] = {
            "check": ("ftn --version",_cray_version_check),
            "build": "ftn -shared -fPIC -h acc,noomp -J {moddir} -o {outpath}"
        }

    builtin_compilers["cray_cpp_openacc"]["CCwrapper"] = {
            "check": ("CC --version",_cray_version_check),
            "build": "CC -shared -fPIC -h acc,noomp -o {outpath}"
        }

    builtin_compilers["cray_fortran_openmp"]["ftnwrapper"] = {
            "check": ("ftn --version",_cray_version_check),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }

    builtin_compilers["cray_cpp_openmp"]["CCwrapper"] = {
            "check": ("CC --version",_cray_version_check),
            "build": "CC -shared -fPIC -fopenmp -o {outpath}"
        }

    builtin_compilers["cray_fortran_fortran"]["ftnwrapper"] = {
            "check": ("ftn --version",_cray_version_check),
            "build": "ftn -shared -fPIC -J {moddir} -o {outpath}"
        }

    builtin_compilers["cray_cpp_cpp"]["CCwrapper"] = {
            "check": ("CC --version",_cray_version_check),
            "build": "CC -shared -fPIC -o {outpath}"
        }

    builtin_compilers["amd_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version",_amd_version_check),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }

    builtin_compilers["amd_cpp_omptarget"]["CCwrapper"] = {
            "check": ("CC --version",_amd_version_check),
            "build": "CC -shared -fPIC -fopenmp -o {outpath}"
        }

    builtin_compilers["amd_fortran_openacc"]["ftnwrapper"] = {
            "check": ("ftn --version",_amd_version_check),
            "build": "ftn -shared -fPIC -fopenacc -J {moddir} -o {outpath}"
        }

    builtin_compilers["amd_cpp_openacc"]["CCwrapper"] = {
            "check": ("CC --version",_amd_version_check),
            "build": "CC -shared -fPIC -fopenacc -o {outpath}"
        }

    builtin_compilers["amd_fortran_openmp"]["ftnwrapper"] = {
            "check": ("ftn --version",_amd_version_check),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
        }

    builtin_compilers["amd_cpp_openmp"]["CCwrapper"] = {
            "check": ("CC --version",_amd_version_check),
            "build": "CC -shared -fPIC -fopenmp -o {outpath}"
        }

    builtin_compilers["amd_fortran_fortran"]["ftnwrapper"] = {
            "check": ("ftn --version",_amd_version_check),
            "build": "ftn -shared -fPIC -J {moddir} -o {outpath}"
        }

    builtin_compilers["amd_cpp_cpp"]["CCwrapper"] = {
            "check": ("CC --version",_amd_version_check),
            "build": "CC -shared -fPIC -o {outpath}"
        }

    builtin_compilers["intel_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version",_intel_version_check),
            "build": "ftn -shared -fpic -qopenacc -module {moddir} -o {outpath}"
        }

    builtin_compilers["intel_fortran_openmp"]["ftnwrapper"] = {
            "check": ("ftn --version",_intel_version_check),
            "build": "ftn -shared -fpic -qopenmp -module {moddir} -o {outpath}"
        }

    builtin_compilers["intel_fortran_fortran"]["ftnwrapper"] = {
            "check": ("ftn --version",_intel_version_check),
            "build": "ftn -shared -fpic -module {moddir} -o {outpath}"
        }

    builtin_compilers["gnu_fortran_omptarget"]["ftnwrapper"] = {
            "check": ("ftn --version", _gnu_version_check),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath} -ffree-line-length-none"
        }

    builtin_compilers["gnu_cpp_omptarget"]["ftnwrapper"] = {
            "check": ("CC --version", _gnu_version_check),
            "build": "CC -shared -fPIC -fopenmp -o {outpath}"
        }

    builtin_compilers["gnu_fortran_openmp"]["ftnwrapper"] = {
            "check": ("ftn --version", _gnu_version_check),
            "build": "ftn -shared -fPIC -fopenmp -J {moddir} -o {outpath} -ffree-line-length-none"
        }

    builtin_compilers["gnu_cpp_openmp"]["ftnwrapper"] = {
            "check": ("CC --version", _gnu_version_check),
            "build": "CC -shared -fPIC -fopenmp -o {outpath}"
        }

    builtin_compilers["gnu_fortran_fortran"]["ftnwrapper"] = {
            "check": ("ftn --version", _gnu_version_check),
            "build": "ftn -shared -fPIC -J {moddir} -o {outpath} -ffree-line-length-none"
        }

    builtin_compilers["gnu_cpp_cpp"]["ftnwrapper"] = {
            "check": ("CC --version", _gnu_version_check),
            "build": "CC -shared -fPIC -J {moddir} -o {outpath}"
        }

builtin_compilers["cray_fortran_omptarget"]["generic"] = {
        "check": ("crayftn --version",_cray_version_check),
        "build": "crayftn -shared -fPIC -h omp,noacc -J {moddir} -o {outpath}"
    }

builtin_compilers["cray_cpp_omptarget"]["generic"] = {
        "check": ("crayCC --version",_cray_version_check),
        "build": "crayCC -shared -fPIC -h omp,noacc -o {outpath}"
    }

builtin_compilers["cray_fortran_openacc"]["generic"] = {
        "check": ("crayftn --version",_cray_version_check),
        "build": "crayftn -shared -fPIC -h acc,noomp -J {moddir} -o {outpath}"
    }

builtin_compilers["cray_cpp_openacc"]["generic"] = {
        "check": ("crayCC --version",_cray_version_check),
        "build": "crayCC -shared -fPIC -h acc,noomp -o {outpath}"
    }

builtin_compilers["cray_fortran_openmp"]["generic"] = {
        "check": ("crayftn --version",_cray_version_check),
        "build": "crayftn -shared -fPIC -h omp,noacc -J {moddir} -o {outpath}"
    }

builtin_compilers["cray_cpp_openmp"]["generic"] = {
        "check": ("crayCC --version",_cray_version_check),
        "build": "crayCC -shared -fPIC -h omp,noacc -o {outpath}"
    }

builtin_compilers["cray_fortran_fortran"]["generic"] = {
        "check": ("crayftn --version",_cray_version_check),
        "build": "crayftn -shared -fPIC -J {moddir} -o {outpath}"
    }

builtin_compilers["cray_cpp_cpp"]["generic"] = {
        "check": ("crayCC --version",_cray_version_check),
        "build": "crayCC -shared -fPIC -o {outpath}"
    }

builtin_compilers["*_cpp_hip"]["generic"] = {
        "check": ("hipcc --version",_hip_check),
        "build": "hipcc -shared -fPIC -lamdhip64 -o {outpath}"
    }

builtin_compilers["amd_fortran_omptarget"]["generic"] = {
        "check": ("amdflang --version",_amd_version_check),
        "build": "amdflang -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
    }

builtin_compilers["amd_cpp_omptarget"]["generic"] = {
        "check": ("amdclang++ --version",_amd_version_check),
        "build": "amdclang++ -shared -fPIC -fopenmp -o {outpath}"
    }

builtin_compilers["amd_fortran_openacc"]["generic"] = {
        "check": ("amdflang --version",_amd_version_check),
        "build": "amdflang -shared -fPIC -fopenacc -J {moddir} -o {outpath}"
    }

builtin_compilers["amd_cpp_openacc"]["generic"] = {
        "check": ("amdclang++ --version",_amd_version_check),
        "build": "amdclang++ -shared -fPIC -fopenacc -o {outpath}"
    }

builtin_compilers["amd_fortran_openmp"]["generic"] = {
        "check": ("amdflang --version",_amd_version_check),
        "build": "amdflang -shared -fPIC -fopenmp -J {moddir} -o {outpath}"
    }

builtin_compilers["amd_cpp_openmp"]["generic"] = {
        "check": ("amdclang++ --version",_amd_version_check),
        "build": "amdclang++ -shared -fPIC -fopenmp -o {outpath}"
    }

builtin_compilers["amd_fortran_fortran"]["generic"] = {
        "check": ("amdflang --version",_amd_version_check),
        "build": "amdflang -shared -fPIC -J {moddir} -o {outpath}"
    }

builtin_compilers["amd_cpp_cpp"]["generic"] = {
        "check": ("amdclang++ --version",_amd_version_check),
        "build": "amdclang++ -shared -fPIC -o {outpath}"
    }

builtin_compilers["intel_fortran_omptarget"]["generic"] = {
        "check": ("ifort --version",_intel_version_check),
        "build": "ifort -shared -fpic -qopenmp -module {moddir} -o {outpath}"
    }

builtin_compilers["intel_cpp_omptarget"]["generic"] = {
        "check": ("icpc --version",_intel_version_check),
        "build": "icpc -shared -fpic -qopenmp -o {outpath}"
    }

builtin_compilers["intel_fortran_openmp"]["generic"] = {
        "check": ("ifort --version",_intel_version_check),
        "build": "ifort -shared -fpic -qopenmp -module {moddir} -o {outpath}"
    }

builtin_compilers["intel_cpp_openmp"]["generic"] = {
        "check": ("icpc --version",_intel_version_check),
        "build": "icpc -shared -fpic -qopenmp -o {outpath}"
    }

builtin_compilers["intel_fortran_fortran"]["generic"] = {
        "check": ("ifort --version",_intel_version_check),
        "build": "ifort -shared -fpic -module {moddir} -o {outpath}"
    }

builtin_compilers["intel_cpp_cpp"]["generic"] = {
        "check": ("icpc --version",_intel_version_check),
        "build": "icpc -shared -fpic -o {outpath}"
    }

builtin_compilers["ibm_fortran_omptarget"]["generic"] = {
        "check": ("xlf_r -qversion",_ibm_version_check),
        "build": "xlf_r -qmkshrobj -qpic -qsmp=omp -qoffload -qmoddir={moddir} -o {outpath}"
    }

builtin_compilers["ibm_cpp_omptarget"]["generic"] = {
        "check": ("xlc++_r -qversion",_ibm_version_check),
        "build": "xlc++_r -qmkshrobj -qpic -qsmp=omp -qoffload -o {outpath}"
    }

builtin_compilers["ibm_fortran_openmp"]["generic"] = {
        "check": ("xlf_r -qversion",_ibm_version_check),
        "build": "xlf_r -qmkshrobj -qpic -qsmp=omp -qmoddir={moddir} -o {outpath}"
    }

builtin_compilers["ibm_cpp_openmp"]["generic"] = {
        "check": ("xlc++_r -qversion",_ibm_version_check),
        "build": "xlc++_r -qmkshrobj -qpic -qsmp=omp -o {outpath}"
    }

builtin_compilers["ibm_fortran_fortran"]["generic"] = {
        "check": ("xlf_r -qversion",_ibm_version_check),
        "build": "xlf_r -qmkshrobj -qpic -qmoddir={moddir} -o {outpath}"
    }

builtin_compilers["ibm_cpp_cpp"]["generic"] = {
        "check": ("xlc++_r -qversion",_ibm_version_check),
        "build": "xlc++_r -qmkshrobj -qpic -o {outpath}"
    }

builtin_compilers["pgi_fortran_omptarget"]["generic"] = {
        "check": ("pgfortran --version", _pgi_version_check),
        "build": "pgfortran -shared -fpic -mp -module {moddir} -o {outpath}"
    }

builtin_compilers["pgi_cpp_omptarget"]["generic"] = {
        "check": ("pgc++ --version", _pgi_version_check),
        "build": "pgc++ -shared -fpic -mp -o {outpath}"
    }


builtin_compilers["pgi_fortran_openacc"]["generic"] = {
        "check": ("pgfortran --version", _pgi_version_check),
        "build": "pgfortran -shared -fpic -acc -module {moddir} -o {outpath}"
    }

builtin_compilers["pgi_cpp_openacc"]["generic"] = {
        "check": ("pgc++ --version", _pgi_version_check),
        "build": "pgc++ -shared -fpic -acc -o {outpath}"
    }

builtin_compilers["pgi_fortran_openmp"]["generic"] = {
        "check": ("pgfortran --version", _pgi_version_check),
        "build": "pgfortran -shared -fpic -mp -module {moddir} -o {outpath}"
    }

builtin_compilers["pgi_cpp_openmp"]["generic"] = {
        "check": ("pgc++ --version", _pgi_version_check),
        "build": "pgc++ -shared -fpic -mp -o {outpath}"
    }

builtin_compilers["pgi_fortran_fortran"]["generic"] = {
        "check": ("pgfortran --version", _pgi_version_check),
        "build": "pgfortran -shared -fpic -module {moddir} -o {outpath}"
    }

builtin_compilers["pgi_cpp_cpp"]["generic"] = {
        "check": ("pgc++ --version", _pgi_version_check),
        "build": "pgc++ -shared -fpic -o {outpath}"
    }

builtin_compilers["*_cpp_cuda"]["generic"] = {
        "check": ("nvcc --version", _cuda_check),
        "build": "nvcc -shared --compiler-options '-fPIC -lcudart' -o {outpath}"
    }

SHLIB = "-dynamiclib" if sys.platform == "darwin" else "-shared"

builtin_compilers["gnu_fortran_omptarget"]["generic"] = {
        "check": ("gfortran --version", _gnu_version_check),
        "build": "gfortran %s -fPIC -fopenmp -J {moddir} -o {outpath} -ffree-line-length-none" % SHLIB
    }

builtin_compilers["gnu_cpp_omptarget"]["generic"] = {
        "check": ("g++ --version", _gnu_version_check),
        "build": "g++ %s -fPIC -fopenmp -o {outpath}" % SHLIB
    }

builtin_compilers["gnu_fortran_openacc"]["generic"] = {
        "check": ("gfortran --version", _gnu_version_check),
        "build": "gfortran %s -fPIC -fopenacc -J {moddir} -o {outpath} -ffree-line-length-none" % SHLIB
    }

builtin_compilers["gnu_cpp_openacc"]["generic"] = {
        "check": ("g++ --version", _gnu_version_check),
        "build": "g++ %s -fPIC -fopenacc -o {outpath}" % SHLIB
    }

builtin_compilers["gnu_fortran_openmp"]["generic"] = {
        "check": ("gfortran --version", _gnu_version_check),
        "build": "gfortran %s -fPIC -fopenmp -J {moddir} -o {outpath} -ffree-line-length-none" % SHLIB
    }

builtin_compilers["gnu_cpp_openmp"]["generic"] = {
        "check": ("g++ --version", _gnu_version_check),
        "build": "g++ %s -fPIC -fopenmp -o {outpath}" % SHLIB
    }

builtin_compilers["gnu_fortran_fortran"]["generic"] = {
        "check": ("gfortran --version", _gnu_version_check),
        "build": "gfortran %s -fPIC -J {moddir} -o {outpath} -ffree-line-length-none" % SHLIB
    }

builtin_compilers["gnu_cpp_cpp"]["generic"] = {
        "check": ("g++ --version", _gnu_version_check),
        "build": "g++ %s -fPIC -o {outpath}" % SHLIB
    }


def build_sharedlib(srcfile, outfile, workdir, compile=None, opts="", vendor=None, lang=None, accel=None):

    srcpath = os.path.join(workdir, srcfile)
    outpath = os.path.join(workdir, outfile)

    if not os.path.isfile(srcpath):
        raise Exception("Source file does not exist: %s" % srcpath)

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
            if _vendor != "*" and _vendor not in vendor: continue

        elif isinstance(vendor, str):
            if _vendor != "*" and _vendor != vendor: continue

        if isinstance(lang, (list, tuple)):
            if _lang != "*" and _lang not in lang: continue

        elif isinstance(lang, str):
            if _lang != "*" and _lang != lang: continue

        if isinstance(accel, (list, tuple)):
            if _accel != "*" and _accel not in accel: continue

        elif isinstance(accel, str):
            if _accel != "*" and _accel != accel: continue
    
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

                print(str(out.stderr).replace("\\n", "\n"))
            except Exception as err:
                print("command fail: %s" % cmd)

    raise Exception("All build commands were failed")
