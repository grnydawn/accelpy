"""accelpy Fortran-based Accelerator module"""

import os

from collections import OrderedDict

from accelpy.util import get_f_dtype
from accelpy.kernel import KernelBase


moddatasrc = """
module modompdata
USE, INTRINSIC :: ISO_C_BINDING

public dataenter, dataexit

contains

INTEGER (C_INT64_T) FUNCTION dataenter({enterargs}) BIND(C, name="dataenter")
    USE, INTRINSIC :: ISO_C_BINDING

    {entervardefs}

    !!$omp target enter data map(to: X(:,:), Y(:,:)) map(alloc: Z(:,:))
    !$omp target enter data map(to: X, Y) map(alloc: Z)

    dataenter = 0

END FUNCTION

INTEGER (C_INT64_T) FUNCTION dataexit({exitargs}) BIND(C, name="dataexit")
    USE, INTRINSIC :: ISO_C_BINDING

    !REAL(C_DOUBLE), DIMENSION(2, 3), INTENT(OUT) :: Z
    {exitvardefs}

    !!$omp target exit data map(from: Z(:,:))
    !$omp target exit data map(from: Z)

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

!    INTEGER i, j
!
!    !$omp target teams num_teams(2)
!    !$omp distribute
!    DO i=1, 2
!        !$omp parallel do
!        DO j=1, 3
!            Z(i, j) = X(i, j) + Y(i, j)
!        END DO
!        !$omp end parallel do
!    END DO
!    !$omp end target teams

    runkernel = 0

END FUNCTION

END MODULE
"""


class FortranKernel(KernelBase):

    lang = "fortran"
    accel = "fortran"


class OpenmpFortranKernel(FortranKernel):
    accel = "openmp"


class OmptargetFortranKernel(OpenmpFortranKernel):
    accel = "omptarget"


    @classmethod
    def _shape(cls, arg):
        return ", ".join([str(s) for s in arg["data"].shape])

    @classmethod
    def _vardefs(cls, arg, intent):
        return "%s, DIMENSION(%s), INTENT(%s) :: %s" % (get_f_dtype(arg),
                cls._shape(arg), intent, arg["curname"])

    @classmethod
    def gen_srcfiles(cls, section, workdir, copyinout, copyin, copyout, alloc):

        datapath = os.path.join(workdir, "omptargetdata.F90")
        kernelpath = os.path.join(workdir, "omptargetkernel.F90")

        dataparams = {}
        kernelparams = {}

        enterargs = []
        exitargs = []
        kernelargs = []

        entervardefs = []
        exitvardefs = []
        kernelvardefs = []

        for cio in copyinout:
            enterargs.append(cio["curname"])
            entervardefs.append(cls._vardefs(cio, "INT"))

            exitargs.append(cio["curname"])
            exitvardefs.append(cls._vardefs(cio, "OUT"))

            kernelargs.append(cio["curname"])
            kernelvardefs.append(cls._vardefs(cio, "INOUT"))

        for ci in copyin:
            enterargs.append(ci["curname"])
            entervardefs.append(cls._vardefs(ci, "IN"))

            kernelargs.append(ci["curname"])
            kernelvardefs.append(cls._vardefs(ci, "IN"))

        for co in copyout:
            enterargs.append(co["curname"])
            entervardefs.append(cls._vardefs(co, "OUT"))

            exitargs.append(co["curname"])
            exitvardefs.append(cls._vardefs(co, "OUT"))

            kernelargs.append(co["curname"])
            kernelvardefs.append(cls._vardefs(co, "OUT"))

        for al in alloc:
            enterargs.append(al["curname"])
            entervardefs.append(cls._vardefs(al, "IN"))

            kernelargs.append(al["curname"])
            kernelvardefs.append(cls._vardefs(co, "IN"))

        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["entervardefs"] = "\n".join(entervardefs)

        dataparams["exitargs"] = ", ".join(exitargs)
        dataparams["exitvardefs"] = "\n".join(exitvardefs)

        kernelparams["kernelargs"] = ", ".join(kernelargs)
        kernelparams["kernelvardefs"] = "\n".join(kernelvardefs)
        kernelparams["kernelbody"] = "\n".join(section.body)

        with open(datapath, "w") as fdata:
            fdata.write(moddatasrc.format(**dataparams))

        with open(kernelpath, "w") as fkernel:
            fkernel.write(modkernelsrc.format(**kernelparams))

        return datapath, kernelpath


_fortaccels = OrderedDict()
KernelBase.avails[FortranKernel.lang] = _fortaccels

_fortaccels[FortranKernel.accel] = FortranKernel
_fortaccels[OpenmpFortranKernel.accel] = OpenmpFortranKernel
_fortaccels[OmptargetFortranKernel.accel] = OmptargetFortranKernel

