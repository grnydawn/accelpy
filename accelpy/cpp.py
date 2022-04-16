"""accelpy Cpp-based Accelerator module"""

import os

from collections import OrderedDict

from accelpy.util import get_c_dtype
from accelpy.accel import AccelBase


convfmt = "{dtype}(*apy_ptr_{name}){shape} = reinterpret_cast<{dtype}(*){shape}>({orgname});"

datasrc = """
#include <stdint.h>

{include}

extern "C" int64_t dataenter_{runid}({enterargs}) {{

    int64_t res;

    {enterdirective}

    res = 0;

    return res;

}}

extern "C" int64_t dataexit_{runid}({exitargs}) {{

    int64_t res;

    {exitdirective}

    res = 0;

    return res;

}}
"""

kernelsrc = """
#include <stdio.h>
#include <stdint.h>

{include}


{macrodefs}

int64_t kernel_{runid}({kernelargs}) {{
    int64_t res;

    {spec}

    res = 0;

    return res;

}}

extern "C" int64_t runkernel_{runid}({runkernelargs}) {{
    int64_t res;

    res = kernel_{runid}({actualargs});

    return res;
}}
"""


class CppAccelBase(AccelBase):

    lang = "cpp"
    srcext = ".cpp"

    @classmethod
    def _gen_include(cls):
        return []

    @classmethod
    def _gen_macrodefs(cls, localvars, modvars):

        typedefs = []
        consts = []
        macros = []

        macros.append("#define TYPE(varname) apy_type_##varname")
        macros.append("#define SHAPE(varname, dim) apy_shape_##varname##dim")
        macros.append("#define SIZE(varname) apy_size_##varname")

        for oldname, arg in modvars:
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            consts.append("const int64_t apy_size_%s = %d;" % (name, arg["data"].size))
            typedefs.append("typedef %s apy_type_%s;" % (dtype, name))

            if ndim > 0:

                shapestr = "".join(["[%d]"%s for s in arg["data"].shape])
                for d, s in enumerate(arg["data"].shape):
                    consts.append("const int64_t apy_shape_%s%d = %d;" % (name, d, s))
            else:
                pass

        for arg in localvars:
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            consts.append("const int64_t apy_size_%s = %d;" % (name, arg["data"].size))
            typedefs.append("typedef %s apy_type_%s;" % (dtype, name))

            if ndim > 0:

                shapestr = "".join(["[%d]"%s for s in arg["data"].shape])
                for d, s in enumerate(arg["data"].shape):
                    consts.append("const int64_t apy_shape_%s%d = %d;" % (name, d, s))
            else:
                pass

        return "\n".join(macros) + "\n\n" + "\n".join(typedefs) + "\n\n" + "\n".join(consts)

    @classmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
                        copyin, copyout, alloc, attr):

        datapath = os.path.join(workdir, filename)

        dataparams = {"runid": str(runid), "datamodname": modname}

        enterargs = []
        exitargs = []
        enterdirective = []
        exitdirective = []

        cionames = []
        cinames = []
        conames = []
        alnames = []

        for cio in copyinout:
            cioname = cio["curname"]
            dtype = get_c_dtype(cio)
            size = cio["data"].size

            if cio["data"].ndim > 0:
                cionames.append("(* %s)[0:%d]" % (cioname, size))
                enterargs.append("%s (* %s)[%d]" % (dtype, cioname, size))
                exitargs.append("%s (* %s)[%d]" % (dtype, cioname, size))

        if cionames:
            enterdirective.append(cls._mapto(cionames))
            exitdirective.append(cls._mapfrom(cionames))

        for ci in copyin:
            ciname = ci["curname"]
            dtype = get_c_dtype(ci)
            size = ci["data"].size

            if ci["data"].ndim > 0:
                cinames.append("(* %s)[0:%d]" % (ciname, size))
                enterargs.append("%s (* %s)[%d]" % (dtype, ciname, size))

        if cinames:
            enterdirective.append(cls._mapto(cinames))

        for co in copyout:
            coname = co["curname"]
            dtype = get_c_dtype(co)
            size = co["data"].size

            if co["data"].ndim > 0:
                conames.append("(* %s)[0:%d]" % (coname, size))
                enterargs.append("%s (* %s)[%d]" % (dtype, coname, size))
                exitargs.append("%s (* %s)[%d]" % (dtype, coname, size))

        if conames:
            alnames.extend(conames)
            exitdirective.append(cls._mapfrom(conames))

        for al in alloc:
            alname = al["curname"]
            dtype = get_c_dtype(al)
            size = al["data"].size

            if al["data"].ndim > 0:
                alnames.append("(* %s)[0:%d]" % (alname, size))
                enterargs.append("%s (* %s)[%d]" % (dtype, alname, size))

        if alnames:
            enterdirective.append(cls._mapalloc(alnames))

        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["enterdirective"] =  "\n".join(enterdirective)
        dataparams["exitargs"] = ", ".join(exitargs)
        dataparams["exitdirective"] = "\n".join(exitdirective)
        dataparams["include"] = "\n".join(cls._gen_include())

        with open(datapath, "w") as fdata:
            fdata.write(datasrc.format(**dataparams))

        #import pdb; pdb.set_trace()
        return datapath

    @classmethod
    def gen_kernelfile(cls, knlhash, dmodname, runid, specid, modattr, section, workdir, localvars, modvars):

        kernelpath = os.path.join(workdir, "K%s%s" % (knlhash[2:], cls.srcext))

        attrspec = section.kwargs.get("attrspec", {})
        modattr.update(attrspec)

        runkernelargs = []
        kernelargs = []
        actualargs = []
        casts = []

        for mname, mvar in modvars:
            ndim = mvar["data"].ndim
            dtype = get_c_dtype(mvar)
            vname = mvar["curname"]

            if ndim > 0:
                shape = "".join(["[%d]"%s for s in mvar["data"].shape])
                runkernelargs.append("%s (* %s)%s" % (dtype, vname, shape))
                kernelargs.append("%s %s%s" % (dtype, vname, shape))
                actualargs.append("*%s" % vname)

            else:
                runkernelargs.append("%s * %s" % (dtype, vname))
                kernelargs.append("%s %s" % (dtype, vname))
                actualargs.append("*%s" % (dtype, vname))

        for lvar in localvars:
            ndim = lvar["data"].ndim
            dtype = get_c_dtype(lvar)
            vname = lvar["curname"]


            if ndim > 0:
                shape = "".join(["[%d]"%s for s in lvar["data"].shape])
                runkernelargs.append("%s (* %s)%s" % (dtype, vname, shape))
                kernelargs.append("%s %s%s" % (dtype, vname, shape))
                actualargs.append("*%s" % vname)

            else:
                runkernelargs.append("%s * %s" % (dtype, vname))
                kernelargs.append("%s %s" % (dtype, vname))
                actualargs.append("*%s" % (dtype, vname))

        kernelparams = {
            "runid": str(runid) + str(specid),
            "macrodefs": cls._gen_macrodefs(localvars, modvars),
            "kernelargs": ", ".join(kernelargs),
            "runkernelargs": ", ".join(runkernelargs),
            "spec": "\n".join(section.body),
            "actualargs":", ".join(actualargs),
            "include":"\n".join(cls._gen_include()) 
        }

        with open(kernelpath, "w") as fkernel:
            fkernel.write(kernelsrc.format(**kernelparams))

        #import pdb; pdb.set_trace()
        return kernelpath

class CppAccel(CppAccelBase):
    accel = "cpp"

    @classmethod
    def _mapto(cls, names):
        return ""

    @classmethod
    def _mapfrom(cls, names):
        return ""

    @classmethod
    def _mapalloc(cls, names):
        return ""


class OpenmpCppAccel(CppAccel):
    accel = "openmp"

    @classmethod
    def _gen_include(cls):
        return ["#include <omp.h>"]


class AcctargetCppAccel(CppAccelBase):
    pass


class OmptargetCppAccel(AcctargetCppAccel):
    accel = "omptarget"

    @classmethod
    def _mapto(cls, names):
        return "#pragma omp target enter data map(to:" + ", ".join(names) + ")"

    @classmethod
    def _mapfrom(cls, names):
        return "#pragma omp target exit data map(from:" + ", ".join(names) + ")"

    @classmethod
    def _mapalloc(cls, names):
        return "#pragma omp target enter data map(alloc:" + ", ".join(names) + ")"


class OpenaccCppAccel(AcctargetCppAccel):
    accel = "openacc"

    @classmethod
    def _mapto(cls, names):
        return "#pragma acc enter data copyin(" + ", ".join(names) + ")"

    @classmethod
    def _mapfrom(cls, names):
        return "#pragma acc exit data copyout(" + ", ".join(names) + ")"

    @classmethod
    def _mapalloc(cls, names):
        return "#pragma acc enter data create(" + ", ".join(names) + ")"


_cppaccels = OrderedDict()
AccelBase.avails[CppAccelBase.lang] = _cppaccels

from .cudahip import CudaAccel, HipAccel

_cppaccels[CudaAccel.accel] = CudaAccel
_cppaccels[HipAccel.accel] = HipAccel

_cppaccels[OmptargetCppAccel.accel] = OmptargetCppAccel
_cppaccels[OpenaccCppAccel.accel] = OpenaccCppAccel
_cppaccels[OpenmpCppAccel.accel] = OpenmpCppAccel
_cppaccels[CppAccel.accel] = CppAccel

