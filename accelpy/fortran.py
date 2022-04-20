"""accelpy Fortran-based Accelerator module"""

import os

from collections import OrderedDict

from accelpy.util import get_f_dtype
from accelpy.accel import AccelBase


moddatasrc = """
module {datamodname}
USE, INTRINSIC :: ISO_C_BINDING

{use}

public dataenter_{runid}, dataexit_{runid}

contains

INTEGER (C_INT64_T) FUNCTION dataenter_{runid}({enterargs}) BIND(C, name="dataenter_{runid}")
    USE, INTRINSIC :: ISO_C_BINDING

    {entervardefs}

    {enterdirective}

    dataenter_{runid} = 0

END FUNCTION

INTEGER (C_INT64_T) FUNCTION dataexit_{runid}({exitargs}) BIND(C, name="dataexit_{runid}")
    USE, INTRINSIC :: ISO_C_BINDING

    {exitvardefs}

    {exitdirective}

    dataexit_{runid} = 0

END FUNCTION

end module
"""

modkernelsrc = """
INTEGER (C_INT64_T) FUNCTION runkernel_{runid}({kernelargs}) BIND(C, name="runkernel_{runid}")
    USE, INTRINSIC :: ISO_C_BINDING

    {use}

    {kernelvardefs}

    {kernelbody}

    runkernel_{runid} = 0

END FUNCTION
"""


class FortranAccelBase(AccelBase):

    lang = "fortran"
    srcext = ".F90"

    @classmethod
    def _gen_include(cls):
        return []

    @classmethod
    def _dimension(cls, arg, attrspec):

        aid = arg["id"]

        if aid in attrspec and "dimension" in attrspec[aid]:
            return attrspec[aid]["dimension"]
                
        return ", ".join([str(s) for s in arg["data"].shape])

    @classmethod
    def _modvardefs(cls, arg):

        dim =  ", ".join([":" for s in arg["data"].shape])

        if dim:
            return "%s, DIMENSION(%s), POINTER :: %s" % (get_f_dtype(arg),
                dim, arg["curname"])

        else:
            return "%s, POINTER :: %s" % (get_f_dtype(arg), arg["curname"])

    @classmethod
    def _funcvardefs(cls, arg, intent, attrspec={}):

        dim = cls._dimension(arg, attrspec)

        if dim:
            return "%s, DIMENSION(%s), INTENT(%s), TARGET :: %s" % (get_f_dtype(arg),
                dim, intent, arg["curname"])

        else:
            return "%s, INTENT(IN) :: %s" % (get_f_dtype(arg), arg["curname"])
#
#    @classmethod
#    def _kernelvardefs(cls, arg, intent, attrspec={}):
#
#        curname = arg["curname"]
#
#        if curname in attrspec and "dimension" in attrspec[curname]:
#            dim = attrspec[curname]["dimension"]
#
#        else:
#            dim = ", ".join([str(s) for s in arg["data"].shape])
#
#        if dim:
#            return "%s, DIMENSION(%s), INTENT(%s), TARGET :: %s" % (get_f_dtype(arg),
#                dim, intent, arg["curname"])
#
#        else:
#            return "%s, INTENT(IN) :: %s" % (get_f_dtype(arg), arg["curname"])

    @classmethod
    def gen_kernelfile(cls, knlhash, dmodname, runid, specid, modattr, section, workdir, localvars, modvars):

        kernelpath = os.path.join(workdir, "K%s%s" % (knlhash[2:], cls.srcext))

        kernelparams = {
            "runid": str(runid) + str(specid)
        }

        kernelargs = []
        kernelvardefs = []

        attrspec = section.kwargs.get("attrspec", {})
        modattr.update(attrspec)

        for old, mvar in modvars:

            kernelargs.append(mvar["curname"])
            kernelvardefs.append(cls._funcvardefs(mvar, "INOUT", attrspec=modattr))

        for lvar in localvars:

            kernelargs.append(lvar["curname"])
            kernelvardefs.append(cls._funcvardefs(lvar, "INOUT", attrspec=attrspec))

        kernelparams["kernelmodname"] = "MOD%s" % knlhash[2:].upper()
        kernelparams["kernelargs"] = ", ".join(kernelargs)
        kernelparams["kernelvardefs"] = "\n".join(kernelvardefs)
        kernelparams["kernelbody"] = "\n".join(section.body)
        kernelparams["use"] = "\n".join(cls._gen_include())

        with open(kernelpath, "w") as fkernel:
            fkernel.write(modkernelsrc.format(**kernelparams))

        #import pdb; pdb.set_trace()
        return kernelpath

    @classmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout, copyin, copyout, alloc, attr):

        datapath = os.path.join(workdir, filename)

        dataparams = {"runid": str(runid), "datamodname": modname}

        enterargs = []
        entervardefs = []
        exitargs = []
        exitvardefs = []

        enterdirective = []
        exitdirective = []

        alnames = []

        cionames = []

        for cio in copyinout:
            cioname = cio["curname"]
            cionames.append(cioname)

            enterargs.append(cioname)
            entervardefs.append(cls._funcvardefs(cio, "INOUT", attrspec=attr))

            exitargs.append(cioname)
            exitvardefs.append(cls._funcvardefs(cio, "INOUT", attrspec=attr))

        if cionames:
            enterdirective.append(cls._mapto(cionames))
            exitdirective.append(cls._mapfrom(cionames))

        cinames = []

        for ci in copyin:
            ciname = ci["curname"]
            cinames.append(ciname)

            enterargs.append(ciname)
            entervardefs.append(cls._funcvardefs(ci, "INOUT", attrspec=attr))

        if cinames:
            enterdirective.append(cls._mapto(cinames))

        conames = []

        for co in copyout:
            coname = co["curname"]
            conames.append(coname)

            enterargs.append(coname)
            entervardefs.append(cls._funcvardefs(co, "INOUT", attrspec=attr))

            exitargs.append(coname)
            exitvardefs.append(cls._funcvardefs(co, "INOUT", attrspec=attr))


        if conames:
            alnames.extend(conames)
            exitdirective.append(cls._mapfrom(conames))

        for al in alloc:
            alname = al["curname"]
            alnames.append(alname)

            enterargs.append(alname)
            entervardefs.append(cls._funcvardefs(al, "INOUT", attrspec=attr))

        if alnames:
            enterdirective.append(cls._mapalloc(alnames))

        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["entervardefs"] = "\n".join(entervardefs)
        dataparams["enterdirective"] = "\n".join(enterdirective)
        dataparams["exitargs"] = ", ".join(exitargs)
        dataparams["exitvardefs"] = "\n".join(exitvardefs)
        dataparams["exitdirective"] = "\n".join(exitdirective)
        dataparams["use"] = "\n".join(cls._gen_include())

        with open(datapath, "w") as fdata:
            fdata.write(moddatasrc.format(**dataparams))

        #import pdb; pdb.set_trace()
        return datapath

class FortranAccel(FortranAccelBase):
    accel = "fortran"

    @classmethod
    def _mapto(cls, names):
        return ""

    @classmethod
    def _mapfrom(cls, names):
        return ""

    @classmethod
    def _mapalloc(cls, names):
        return ""


class OpenmpFortranAccel(FortranAccel):
    accel = "openmp"

    @classmethod
    def _gen_include(cls):
        return ["USE OMP_LIB"]


class AcctargetFortranAccel(FortranAccelBase):
    pass


class OmptargetFortranAccel(AcctargetFortranAccel):
    accel = "omptarget"

    @classmethod
    def _gen_include(cls):
        return ["USE OMP_LIB"]

    @classmethod
    def _mapto(cls, names):
        return "!$omp target enter data map(to:" + ", ".join(names) + ")"

    @classmethod
    def _mapfrom(cls, names):
        return "!$omp target exit data map(from:" + ", ".join(names) + ")"

    @classmethod
    def _mapalloc(cls, names):
        return "!$omp target enter data map(alloc:" + ", ".join(names) + ")"


class OpenaccFortranAccel(AcctargetFortranAccel):
    accel = "openacc"

    @classmethod
    def _mapto(cls, names):
        return "!$acc enter data copyin(" + ", ".join(names) + ")"

    @classmethod
    def _mapfrom(cls, names):
        return "!$acc exit data copyout(" + ", ".join(names) + ")"

    @classmethod
    def _mapalloc(cls, names):
        return "!$acc enter data create(" + ", ".join(names) + ")"

_fortaccels = OrderedDict()
AccelBase.avails[FortranAccelBase.lang] = _fortaccels

_fortaccels[OmptargetFortranAccel.accel] = OmptargetFortranAccel
_fortaccels[OpenaccFortranAccel.accel] = OpenaccFortranAccel
_fortaccels[OpenmpFortranAccel.accel] = OpenmpFortranAccel
_fortaccels[FortranAccel.accel] = FortranAccel

