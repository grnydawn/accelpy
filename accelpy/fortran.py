"""accelpy Fortran-based Accelerator module"""

import os, uuid, sys

from collections import OrderedDict

from accelpy.util import get_f_dtype
from accelpy.accel import AccelBase


moddatasrc = """
module {datamodname}
USE, INTRINSIC :: ISO_C_BINDING

{modvardefs}

public dataenter_{runid}, dataexit_{runid}
public {modvars}

contains

INTEGER (C_INT64_T) FUNCTION dataenter_{runid}({enterargs}) BIND(C, name="dataenter_{runid}")
    USE, INTRINSIC :: ISO_C_BINDING

    {entervardefs}

    {enterassign}

    {enterdirective}

    dataenter_{runid} = 0

END FUNCTION

INTEGER (C_INT64_T) FUNCTION dataexit_{runid}() BIND(C, name="dataexit_{runid}")
    USE, INTRINSIC :: ISO_C_BINDING

    {exitdirective}

    dataexit_{runid} = 0

END FUNCTION

end module
"""

modkernelsrc = """
!MODULE {kernelmodname}

!public runkernel_{runid}

!CONTAINS

INTEGER (C_INT64_T) FUNCTION runkernel_{runid}({kernelargs}) BIND(C, name="runkernel_{runid}")
    USE, INTRINSIC :: ISO_C_BINDING
    {useonlyvars}

    {kernelvardefs}

    {kernelbody}

    runkernel_{runid} = 0

END FUNCTION

!END MODULE
"""


class FortranAccelBase(AccelBase):

    lang = "fortran"
    srcext = ".F90"
    libext = ".dylib" if sys.platform == "darwin" else ".so"


class FortranAccel(FortranAccelBase):
    accel = "fortran"

    def _mapto(cls, names):
        raise NotImplementedError("_mapto")

    def _mapfrom(cls, names):
        raise NotImplementedError("_mapfrom")

    def _mapalloc(cls, names):
        raise NotImplementedError("_mapalloc")

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
    def _entervardefs(cls, arg, intent, attrspec={}):

        dim = cls._dimension(arg, attrspec)

        if dim:
            return "%s, DIMENSION(%s), INTENT(%s), TARGET :: %s" % (get_f_dtype(arg),
                dim, intent, "l"+arg["curname"])

        else:
            return "%s, INTENT(IN) :: %s" % (get_f_dtype(arg), arg["curname"])

    @classmethod
    def _kernelvardefs(cls, arg, intent, attrspec={}):

        curname = arg["curname"]

        if curname in attrspec and "dimension" in attrspec[curname]:
            return attrspec[curname]["dimension"]
                
        dim = ", ".join([str(s) for s in arg["data"].shape])

        if dim:
            return "%s, DIMENSION(%s), INTENT(%s), TARGET :: %s" % (get_f_dtype(arg),
                dim, intent, "l"+arg["curname"])

        else:
            return "%s, INTENT(IN) :: %s" % (get_f_dtype(arg), arg["curname"])

    @classmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
                        copyin, copyout, alloc, attr):

        datapath = os.path.join(workdir, filename)

        dataparams = {"runid": str(runid), "datamodname": modname}

        modvardefs = []
        modvars = []

        enterargs = []
        entervardefs = []
        enterassign = []

        for cio in copyinout:
            cioname = cio["curname"]

            modvars.append(cioname)
            enterargs.append("l"+cioname)
            entervardefs.append(cls._entervardefs(cio, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(cio))
            enterassign.append("%s => %s" % (cioname, "l"+cioname))

        for ci in copyin:
            ciname = ci["curname"]

            modvars.append(ciname)
            enterargs.append("l"+ciname)
            entervardefs.append(cls._entervardefs(ci, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(ci))
            enterassign.append("%s => %s" % (ciname, "l"+ciname))

        for co in copyout:
            coname = co["curname"]

            modvars.append(coname)
            enterargs.append("l"+coname)
            entervardefs.append(cls._entervardefs(co, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(co))
            enterassign.append("%s => %s" % (coname, "l"+coname))

        for al in alloc:
            alname = al["curname"]

            modvars.append(alname)
            enterargs.append("l"+alname)
            entervardefs.append(cls._entervardefs(al, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(al))
            enterassign.append("%s => %s" % (alname, "l"+alname))

        dataparams["modvardefs"] = "\n".join(modvardefs)
        dataparams["modvars"] = ", ".join(modvars)
        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["entervardefs"] = "\n".join(entervardefs)
        dataparams["enterdirective"] = ""
        dataparams["enterassign"] = "\n".join(enterassign)
        dataparams["exitdirective"] = ""

        with open(datapath, "w") as fdata:
            fdata.write(moddatasrc.format(**dataparams))

        return datapath

    @classmethod
    def gen_kernelfile(cls, knlhash, dmodname, runid, section, workdir, localvars, modvars):

        kernelpath = os.path.join(workdir, "K%s%s" % (knlhash[2:], cls.srcext))

        kernelparams = {"runid": str(runid)}
        kernelargs = []
        uonlyvars = []
        kernelvardefs = []

        for old, new in modvars:
            uonlyvars.append("USE %s, ONLY : %s => %s" % (dmodname, new, old))

        attrspec = section.kwargs.get("attrspec", {})

        for lvar in localvars:

            kernelargs.append(lvar["curname"])
            kernelvardefs.append(cls._kernelvardefs(lvar, "INOUT", attrspec=attrspec))

        kernelparams["kernelmodname"] = "MOD%s" % knlhash[2:].upper()
        kernelparams["kernelargs"] = ", ".join(kernelargs)
        kernelparams["useonlyvars"] = "\n".join(uonlyvars)
        kernelparams["kernelvardefs"] = "\n".join(kernelvardefs)
        kernelparams["kernelbody"] = "\n".join(section.body)

        with open(kernelpath, "w") as fkernel:
            fkernel.write(modkernelsrc.format(**kernelparams))

        #import pdb; pdb.set_trace()
        return kernelpath


class OpenmpFortranAccel(FortranAccel):
    accel = "openmp"


class AcctargetFortranAccel(FortranAccelBase):

    def _mapto(cls, names):
        raise NotImplementedError("_mapto")

    def _mapfrom(cls, names):
        raise NotImplementedError("_mapfrom")

    def _mapalloc(cls, names):
        raise NotImplementedError("_mapalloc")

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
    def _entervardefs(cls, arg, intent, attrspec={}):

        dim = cls._dimension(arg, attrspec)

        if dim:
            return "%s, DIMENSION(%s), INTENT(%s), TARGET :: %s" % (get_f_dtype(arg),
                dim, intent, "l"+arg["curname"])

        else:
            return "%s, INTENT(IN) :: %s" % (get_f_dtype(arg), arg["curname"])

    @classmethod
    def _kernelvardefs(cls, arg, intent, attrspec={}):

        curname = arg["curname"]

        if curname in attrspec and "dimension" in attrspec[curname]:
            return attrspec[curname]["dimension"]
                
        dim = ", ".join([str(s) for s in arg["data"].shape])

        if dim:
            return "%s, DIMENSION(%s), INTENT(%s), TARGET :: %s" % (get_f_dtype(arg),
                dim, intent, "l"+arg["curname"])

        else:
            return "%s, INTENT(IN) :: %s" % (get_f_dtype(arg), arg["curname"])

    @classmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout, copyin, copyout, alloc, attr):

        datapath = os.path.join(workdir, filename)

        dataparams = {"runid": str(runid), "datamodname": modname}

        modvardefs = []
        modvars = []

        enterargs = []
        entervardefs = []
        enterassign = []

        enterdirective = []
        exitdirective = []

        alnames = []

        cionames = []

        for cio in copyinout:
            cioname = cio["curname"]
            cionames.append(cioname)

            modvars.append(cioname)
            enterargs.append("l"+cioname)
            entervardefs.append(cls._entervardefs(cio, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(cio))
            enterassign.append("%s => %s" % (cioname, "l"+cioname))

        if cionames:
            #enterdirective.append("!$omp target enter data map(to:" +
            #                        ", ".join(cionames) + ")")
            enterdirective.append(cls._mapto(cionames))
            #exitdirective.append("!$omp target exit data map(from:" +
            #                        ", ".join(cionames) + ")")
            exitdirective.append(cls._mapfrom(cionames))

        cinames = []

        for ci in copyin:
            ciname = ci["curname"]
            cinames.append(ciname)

            modvars.append(ciname)
            enterargs.append("l"+ciname)
            entervardefs.append(cls._entervardefs(ci, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(ci))
            enterassign.append("%s => %s" % (ciname, "l"+ciname))

        if cinames:
            #enterdirective.append("!$omp target enter data map(to:" +
            #                        ", ".join(cinames) + ")")
            enterdirective.append(cls._mapto(cinames))

        conames = []

        for co in copyout:
            coname = co["curname"]
            conames.append(coname)

            modvars.append(coname)
            enterargs.append("l"+coname)
            entervardefs.append(cls._entervardefs(co, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(co))
            enterassign.append("%s => %s" % (coname, "l"+coname))


        if conames:
            alnames.extend(conames)
            #exitdirective.append("!$omp target exit data map(from:" +
            #                        ", ".join(conames) + ")")
            exitdirective.append(cls._mapfrom(conames))

        for al in alloc:
            alname = al["curname"]
            alnames.append(alname)

            modvars.append(alname)
            enterargs.append("l"+alname)
            entervardefs.append(cls._entervardefs(al, "INOUT", attrspec=attr))
            modvardefs.append(cls._modvardefs(al))
            enterassign.append("%s => %s" % (alname, "l"+alname))

        if alnames:
            #enterdirective.append("!$omp target enter data map(alloc:" +
            #                        ", ".join(alnames) + ")")
            enterdirective.append(cls._mapalloc(alnames))

        dataparams["modvardefs"] = "\n".join(modvardefs)
        dataparams["modvars"] = ", ".join(modvars)
        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["entervardefs"] = "\n".join(entervardefs)
        dataparams["enterdirective"] = "\n".join(enterdirective)
        dataparams["enterassign"] = "\n".join(enterassign)

        dataparams["exitdirective"] = "\n".join(exitdirective)

        with open(datapath, "w") as fdata:
            fdata.write(moddatasrc.format(**dataparams))

        return datapath

    @classmethod
    def gen_kernelfile(cls, knlhash, dmodname, runid, section, workdir, localvars, modvars):

        kernelpath = os.path.join(workdir, "K%s%s" % (knlhash[2:], cls.srcext))

        kernelparams = {"runid": str(runid)}
        kernelargs = []
        uonlyvars = []
        kernelvardefs = []

        for old, new in modvars:
            uonlyvars.append("USE %s, ONLY : %s => %s" % (dmodname, new, old))

        attrspec = section.kwargs.get("attrspec", {})

        for lvar in localvars:

            kernelargs.append(lvar["curname"])
            kernelvardefs.append(cls._kernelvardefs(lvar, "INOUT", attrspec=attrspec))

        kernelparams["kernelmodname"] = "MOD%s" % knlhash[2:].upper()
        kernelparams["kernelargs"] = ", ".join(kernelargs)
        kernelparams["useonlyvars"] = "\n".join(uonlyvars)
        kernelparams["kernelvardefs"] = "\n".join(kernelvardefs)
        kernelparams["kernelbody"] = "\n".join(section.body)

        with open(kernelpath, "w") as fkernel:
            fkernel.write(modkernelsrc.format(**kernelparams))

        #import pdb; pdb.set_trace()
        return kernelpath

class OmptargetFortranAccel(AcctargetFortranAccel):
    accel = "omptarget"

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

_fortaccels[FortranAccel.accel] = FortranAccel
_fortaccels[OpenmpFortranAccel.accel] = OpenmpFortranAccel
_fortaccels[OmptargetFortranAccel.accel] = OmptargetFortranAccel
_fortaccels[OpenaccFortranAccel.accel] = OpenaccFortranAccel

