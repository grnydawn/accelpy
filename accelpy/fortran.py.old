"""accelpy Fortran Accelerator module"""
  

from accelpy.accel import AccelBase, get_compilers
from ctypes import c_int, c_longlong, c_float, c_double

t_main = """
{typeattr}

{commonblock}

{testcode}

{datacopies}

{attrproc}

INTEGER (C_INT) FUNCTION accelpy_start(device, channel) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE
    INTEGER (C_INT), INTENT(IN) :: device, channel

    accelpy_start = accelpy_kernel(device, channel)

CONTAINS

{kernel}

END FUNCTION

INTEGER (C_INT) FUNCTION accelpy_stop() BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    accelpy_stop = 0

END FUNCTION

INTEGER (C_INT) FUNCTION accelpy_isbusy() BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    accelpy_isbusy = 0

END FUNCTION
"""

t_typeattr = """
TYPE :: accelpy_attrtype
    USE, INTRINSIC :: ISO_C_BINDING
    INTEGER (C_INT), DIMENSION(:), ALLOCATABLE :: attrs

CONTAINS

    PROCEDURE :: ndim => accelpy_ndim
    PROCEDURE :: itemsize => accelpy_itemsize
    PROCEDURE :: size => accelpy_size
    PROCEDURE :: shape => accelpy_shape
    PROCEDURE :: stride => accelpy_stride
    PROCEDURE :: unravel_index => accelpy_unravel_index
END TYPE
"""

t_common = """
BLOCK DATA SETUP
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    {commonvars}
END
"""

t_attrproc = """
INTEGER (C_INT) FUNCTION accelpy_ndim(self)
    CLASS(accelpy_attrtype), INTENT(IN) :: self

    ndim = self%attrs(1)
END FUNCTION
INTEGER (C_INT) FUNCTION accelpy_itemsize(self)
    CLASS(accelpy_attrtype), INTENT(IN) :: self

    itemsize = self%attrs(2)
END FUNCTION

INTEGER (C_INT) FUNCTION accelpy_size(self)
    CLASS(accelpy_attrtype), INTENT(IN) :: self

    size = self%attrs(3)
END FUNCTION

INTEGER (C_INT) FUNCTION accelpy_shape(self, dim)
    CLASS(accelpy_attrtype), INTENT(IN) :: self
    INTEGER, INTENT(IN) :: dim

    shape = self%attrs(3+dim)
END FUNCTION

INTEGER (C_INT) FUNCTION accelpy_stride(self, dim)
    CLASS(accelpy_attrtype), INTENT(IN) :: self
    INTEGER, INTENT(IN) :: dim

    stride = self%attrs(3+self%attrs(1)+dim)
END FUNCTION

INTEGER (C_INT) FUNCTION accelpy_unravel_index(self, tid, dim)
    CLASS(accelpy_attrtype), INTENT(IN) :: self
    INTEGER, INTENT(IN) :: dim, tid
    INTEGER :: i
    INTEGER (C_INT) :: q, r, s

    r = tid-1

    DO i=0,dim-1
        s = self%stride(i+1)
        q = r / s
        r = MOD(r, s)
    END DO

    unravel_index = q+1
END FUNCTION
"""

t_h2a = """
INTEGER (C_INT) FUNCTION {funcname} (data, attrs, attrsize_) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data
    INTEGER (C_INT), DIMENSION(*), INTENT(IN) :: attrs
    INTEGER (C_INT), INTENT(IN) :: attrsize_
    {common}

    {varname} => data
    ALLOCATE({varname}_attr%attrs(attrsize_))
    {varname}_attr%attrs(:) = attrs(1:attrsize_)

    {funcname} = 0

END FUNCTION
"""

t_a2hcopy = """
INTEGER (C_INT) FUNCTION {funcname} (data) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(OUT) :: data

    {funcname} = 0

END FUNCTION
"""

t_testfunc = """
INTEGER (C_INT) FUNCTION accelpy_test_run()
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE
    COMMON /accelpy/ {varin}, {varout}
    COMMON /accelpy/ {varin}_attr
    INTEGER :: id

    DO id=1, {varin}_attr%shape(1)
        {varout}(id) = {varin}(id)
    END DO

    accelpy_test_run = 0

END FUNCTION
"""

t_kernel = """
INTEGER (C_INT) FUNCTION accelpy_kernel(device, channel)
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE
    INTEGER (C_INT), INTENT(IN) :: device, channel
    {commonvars}

    INTEGER :: ACCELPY_WORKER_ID = 0

    {order}

    accelpy_kernel = 0
end FUNCTION
"""

class FortranAccel(AccelBase):

    name = "fortran"
    lang = "fortran"

    dtypemap = {
        "int32": ["INTEGER (C_INT)", c_int],
        "int64": ["INTEGER (C_LONG)", c_longlong],
        "float32": ["REAL (C_FLOAT)", c_float],
        "float64": ["REAL (C_DOUBLE)", c_double]
    }

    def gen_typeattr(self):

        return t_typeattr

    def _commonvar(self, arg):

        ndim = arg["data"].ndim
        dtype = self.get_dtype(arg)

        dimension = (("DIMENSION("+",".join([":"]*ndim)+"), POINTER")
                        if ndim > 0 else "")

        out = "%s %s :: %s\n" % (dtype, dimension, arg["curname"])
        out += "COMMON /accelpy/ %s\n" % arg["curname"]

        return out

    def _commonattr(self, arg):

        ndim = arg["data"].ndim
        dtype = self.get_dtype(arg)

        dimension = (("DIMENSION("+",".join([":"]*ndim)+"), POINTER")
                        if ndim > 0 else "")

        out = "CLASS(accelpy_attrtype) :: %s_attr\n" % arg["curname"]
        out += "COMMON /accelpy/ %s_attr\n" % arg["curname"]

        return out

    def gen_commonblock(self, inputs, outputs):

        common = []

        for arg in self._testdata+inputs+outputs:
#
#            ndim = arg["data"].ndim
#            dtype = self.get_dtype(arg)
#
#            dimension = (("DIMENSION("+",".join([":"]*ndim)+"), POINTER")
#                            if ndim > 0 else "")
#
#            common.append("%s %s :: %s" % (dtype, dimension, arg["curname"]))
#            common.append("CLASS(accelpy_attrtype) :: %s_attr" % arg["curname"])
#            common.append("COMMON /accelpy/ %s" % arg["curname"])
#            common.append("COMMON /accelpy/ %s_attr" % arg["curname"])
            common.append(self._commonvar(arg))
            common.append(self._commonattr(arg))
            common.append("\n")

        return t_common.format(commonvars="\n".join(common))

    def gen_testcode(self):

        out = []

        input = self._testdata[0]
        output = self._testdata[1]

        bound_in = ",".join([":"]*input["data"].ndim)
        bound_out = ",".join([":"]*output["data"].ndim)
        vartype_in = self.get_vartype(input)
        vartype_out = self.get_vartype(output)
        dtype_in = self.get_dtype(input)
        dtype_out = self.get_dtype(input)
        funcname_in = "accelpy_test_h2acopy"
        funcname_out = "accelpy_test_h2amalloc"
        common_in = self._commonvar(input)+self._commonattr(input)
        common_out = self._commonvar(output)+self._commonattr(output)
        funcname_a2h = "accelpy_test_a2hcopy"

        out.append(t_h2a.format(funcname=funcname_in, bound=bound_in,
                    varname=input["curname"], common=common_in, dtype=dtype_in))
        out.append(t_h2a.format(funcname=funcname_out, bound=bound_out,
                    varname=output["curname"], common=common_out, dtype=dtype_out))
        out.append(t_a2hcopy.format(funcname=funcname_a2h,
                    dtype=dtype_out, bound=bound_out))

        out.append(t_testfunc.format(varin=input["curname"],
                    varout=output["curname"]))

        return "\n".join(out)

    def gen_datacopies(self, inputs, outputs):

        out = []

        for input in inputs:
            vartype = self.get_vartype(input)
            dtype = self.get_dtype(input)
            funcname = self.getname_h2acopy(input)
            bound = ",".join([":"]*input["data"].ndim)
            common = self._commonvar(input)+self._commonattr(input)

            out.append(t_h2a.format(funcname=funcname, bound=bound,
                        varname=input["curname"], common=common, dtype=dtype))

        for output in outputs:
            vartype = self.get_vartype(output)
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)
            bound = ",".join([":"]*output["data"].ndim)
            common = self._commonvar(output)+self._commonattr(output)

            out.append(t_h2a.format(funcname=funcname, bound=bound,
                        varname=output["curname"], common=common, dtype=dtype))

        for output in outputs:
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)
            bound = ",".join([":"]*output["data"].ndim)

            out.append(t_a2hcopy.format(funcname=funcname, bound=bound,
                        dtype=dtype))

        return "\n".join(out)

    def gen_kernel(self, inputs, outputs):

        order =  self._order.get_section(self.name)
        common = []

        for arg in inputs+outputs:
            common.append(self._commonvar(arg))
            common.append(self._commonattr(arg))

        return t_kernel.format(order="\n".join(order.body),
                        commonvars="\n".join(common))

    def gen_attrproc(self):

        return t_attrproc

    def gen_code(self, inputs, outputs):

        main_fmt = {
            "typeattr":self.gen_typeattr(),
            "commonblock":self.gen_commonblock(inputs, outputs),
            "testcode":self.gen_testcode(),
            "datacopies":self.gen_datacopies(inputs, outputs),
            "kernel":self.gen_kernel(inputs, outputs),
            "attrproc":self.gen_attrproc(),
        }

        code = t_main.format(**main_fmt)

        return code

    def getname_h2amalloc(self, arg):
        return "accelpy_h2amalloc_%s" % arg["curname"]

    def getname_h2acopy(self, arg):
        return "accelpy_h2acopy_%s" % arg["curname"]

    def getname_a2hcopy(self, arg):
        return "accelpy_a2hcopy_%s" % arg["curname"]


AccelBase.avails[FortranAccel.name] = FortranAccel
