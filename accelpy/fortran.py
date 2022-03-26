"""accelpy Fortran-based Accelerator module"""

import os, uuid

from collections import OrderedDict

from accelpy.util import get_f_dtype
from accelpy.accel import AccelBase


moddatasrc = """
module modompdata
USE, INTRINSIC :: ISO_C_BINDING

public dataenter, dataexit

contains

INTEGER (C_INT64_T) FUNCTION dataenter({enterargs}) BIND(C, name="dataenter")
    USE, INTRINSIC :: ISO_C_BINDING

    {entervardefs}

    {enterdirective}

    dataenter = 0

END FUNCTION

INTEGER (C_INT64_T) FUNCTION dataexit({exitargs}) BIND(C, name="dataexit")
    USE, INTRINSIC :: ISO_C_BINDING

    {exitvardefs}

    {exitdirective}

    dataexit = 0

END FUNCTION

end module
"""

modkernelsrc = """
MODULE modompkernel

public runkernel

CONTAINS

INTEGER (C_INT64_T) FUNCTION runkernel({kernelargs}) BIND(C, name="runkernel")
    USE, INTRINSIC :: ISO_C_BINDING

    {kernelvardefs}

    {kernelbody}

    runkernel = 0

END FUNCTION

END MODULE
"""


class FortranAccel(AccelBase):

    lang = "fortran"
    accel = "fortran"


class OpenmpFortranAccel(FortranAccel):
    accel = "openmp"


class OmptargetFortranAccel(OpenmpFortranAccel):
    accel = "omptarget"

    @classmethod
    def _dimension(cls, arg, attrspec):

        curname = arg["curname"]

        if curname in attrspec and "dimension" in attrspec[curname]:
            return attrspec[curname]["dimension"]
                
        return ", ".join([str(s) for s in arg["data"].shape])

    @classmethod
    def _vardefs(cls, arg, intent, attrspec={}):

        dim = cls._dimension(arg, attrspec)
        if dim:
            return "%s, DIMENSION(%s), INTENT(%s) :: %s" % (get_f_dtype(arg),
                dim, intent, arg["curname"])

        else:
            return "%s, INTENT(%s) :: %s" % (get_f_dtype(arg), intent,
                arg["curname"])

    @classmethod
    def gen_datafile(cls, workdir, copyinout, copyin, copyout, alloc):

        filename = uuid.uuid4().hex[:10]

        datapath = os.path.join(workdir, "data-%s.F90" % filename)

        dataparams = {}

        enterargs = []
        exitargs = []

        entervardefs = []
        exitvardefs = []

        enterdirective = []
        exitdirective = []

        alnames = []

        cionames = []

        for cio in copyinout:
            cioname = cio["curname"]
            cionames.append(cioname)

            enterargs.append(cioname)
            entervardefs.append(cls._vardefs(cio, "IN"))

            exitargs.append(cioname)
            exitvardefs.append(cls._vardefs(cio, "OUT"))

        if cionames:
            enterdirective.append("!$omp target enter data map(to:" +
                                    ", ".join(cionames) + ")")
            exitdirective.append("!$omp target exit data map(from:" +
                                    ", ".join(cionames) + ")")

        cinames = []

        for ci in copyin:
            ciname = ci["curname"]
            cinames.append(ciname)

            enterargs.append(ciname)
            entervardefs.append(cls._vardefs(ci, "IN"))

        if cinames:
            enterdirective.append("!$omp target enter data map(to:" +
                                    ", ".join(cinames) + ")")

        conames = []

        for co in copyout:
            coname = co["curname"]
            conames.append(coname)

            enterargs.append(coname)
            entervardefs.append(cls._vardefs(co, "IN"))

            exitargs.append(coname)
            exitvardefs.append(cls._vardefs(co, "OUT"))

        if conames:
            alnames.extend(conames)
            exitdirective.append("!$omp target exit data map(from:" +
                                    ", ".join(conames) + ")")


        for al in alloc:
            alname = al["curname"]
            alnames.append(alname)

            enterargs.append(alname)
            entervardefs.append(cls._vardefs(al, "IN"))

        if alnames:
            enterdirective.append("!$omp target enter data map(alloc:" +
                                    ", ".join(alnames) + ")")

        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["entervardefs"] = "\n".join(entervardefs)
        dataparams["enterdirective"] = "\n".join(enterdirective)

        dataparams["exitargs"] = ", ".join(exitargs)
        dataparams["exitvardefs"] = "\n".join(exitvardefs)
        dataparams["exitdirective"] = "\n".join(exitdirective)

        with open(datapath, "w") as fdata:
            fdata.write(moddatasrc.format(**dataparams))

        #import pdb; pdb.set_trace()
        return datapath

    @classmethod
    def gen_kernelfile(cls, section, workdir, localvars):

        filename = uuid.uuid4().hex[:10]

        kernelpath = os.path.join(workdir, "kernel-%s.F90" % filename)

        kernelparams = {}
        kernelargs = []
        kernelvardefs = []

        attrspec = section.kwargs.get("attrspec", {})

        for lvar in localvars:

            kernelargs.append(lvar["curname"])
            kernelvardefs.append(cls._vardefs(lvar, "INOUT", attrspec=attrspec))

        kernelparams["kernelargs"] = ", ".join(kernelargs)
        kernelparams["kernelvardefs"] = "\n".join(kernelvardefs)
        kernelparams["kernelbody"] = "\n".join(section.body)

        with open(kernelpath, "w") as fkernel:
            fkernel.write(modkernelsrc.format(**kernelparams))

        #import pdb; pdb.set_trace()
        return kernelpath


_fortaccels = OrderedDict()
AccelBase.avails[FortranAccel.lang] = _fortaccels

_fortaccels[FortranAccel.accel] = FortranAccel
_fortaccels[OpenmpFortranAccel.accel] = OpenmpFortranAccel
_fortaccels[OmptargetFortranAccel.accel] = OmptargetFortranAccel

