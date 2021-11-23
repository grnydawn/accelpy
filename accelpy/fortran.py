"""accelpy Fortran Accelerator module"""


from accelpy.accel import AccelBase, get_compilers
from ctypes import c_int, c_longlong, c_float, c_double

t_main = """

{testcode}

"""

t_module = """
MODULE accelpy_global
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

{typeattr}

{testvars}

CONTAINS

{procattr}

END MODULE
"""

t_h2a = """
!INTEGER (C_LONG_LONG) FUNCTION {funcname} (data, attrs, attrsize_) BIND(C)
INTEGER (C_LONG_LONG) FUNCTION {funcname} (data, attrs) BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}, {varname}_attr
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data
    !INTEGER (C_INT), DIMENSION(*), INTENT(IN) :: attrs
    !INTEGER (C_INT), INTENT(IN) :: attrsize_
    INTEGER (C_LONG_LONG), DIMENSION(*), INTENT(IN) :: attrs
    !INTEGER (C_LONG_LONG), INTENT(IN) :: attrsize_

    {varname} => data
    ALLOCATE({varname}_attr)
    !print *, "TTT", attrs(1), attrs(3), attrs(5)
    !ALLOCATE({varname}_attr%attrs(attrsize_))
    !{varname}_attr%attrs(:) = attrs(1:attrsize_)

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
    USE accelpy_global, ONLY : {varin}, {varout},{varin}_attr
    IMPLICIT NONE
    INTEGER :: id

    DO id=1, {varin}_attr%shape(1)
        {varout}(id) = {varin}(id)
    END DO

    accelpy_test_run = 0

END FUNCTION
"""

t_typeattr = """
TYPE :: accelpy_attrtype
    INTEGER (C_lONG_LONG), DIMENSION(:), ALLOCATABLE :: attrs

CONTAINS

    PROCEDURE :: ndim => accelpy_ndim
    PROCEDURE :: itemsize => accelpy_itemsize
    PROCEDURE :: size => accelpy_size
    PROCEDURE :: shape => accelpy_shape
    PROCEDURE :: stride => accelpy_stride
    PROCEDURE :: unravel_index => accelpy_unravel_index
END TYPE
"""

t_procattr = """
INTEGER (C_INT) FUNCTION accelpy_ndim(self)
    CLASS(accelpy_attrtype), INTENT(IN) :: self

    accelpy_ndim = self%attrs(1)
END FUNCTION
INTEGER (C_INT) FUNCTION accelpy_itemsize(self)
    CLASS(accelpy_attrtype), INTENT(IN) :: self

    accelpy_itemsize = self%attrs(2)
END FUNCTION

INTEGER (C_INT) FUNCTION accelpy_size(self)
    CLASS(accelpy_attrtype), INTENT(IN) :: self

    accelpy_size = self%attrs(3)
END FUNCTION

INTEGER (C_INT) FUNCTION accelpy_shape(self, dim)
    CLASS(accelpy_attrtype), INTENT(IN) :: self
    INTEGER, INTENT(IN) :: dim

    accelpy_shape = self%attrs(3+dim)
END FUNCTION

INTEGER (C_INT) FUNCTION accelpy_stride(self, dim)
    CLASS(accelpy_attrtype), INTENT(IN) :: self
    INTEGER, INTENT(IN) :: dim

    accelpy_stride = self%attrs(3+self%attrs(1)+dim)
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

    accelpy_unravel_index = q+1
END FUNCTION
"""

class FortranAccel(AccelBase):

    name = "fortran"
    lang = "fortran"

    dtypemap = {
        "int32": ["INTEGER (C_INT)", c_int],
        "int64": ["INTEGER (C_LONG_LONG)", c_longlong],
        "float32": ["REAL (C_FLOAT)", c_float],
        "float64": ["REAL (C_DOUBLE)", c_double]
    }

    def gen_code(self, inputs, outputs):

        module_fmt = {
            "typeattr": self._get_typeattr(),
            "testvars": self._get_testvars(),
            "procattr": self._get_procattr(),
        }
        module = t_module.format(**module_fmt)

        main_fmt = {
            "testcode": self._get_testcode()
        }
        main = t_main.format(**main_fmt)

        return [module, main]

    def _get_procattr(self):

        return t_procattr

    def _get_typeattr(self):

        return t_typeattr

    def _get_testvars(self):

        out = []

        for arg in self._testdata:
            ndim = arg["data"].ndim
            dtype = self.get_dtype(arg)

            dimension = (("DIMENSION("+",".join([":"]*ndim)+"), POINTER")
                            if ndim > 0 else "")

            out.append("%s, %s :: %s" % (dtype, dimension, arg["curname"]))
            out.append("CLASS(accelpy_attrtype), ALLOCATABLE :: %s_attr" % arg["curname"])

        return "\n".join(out)

    def _get_testcode(self):

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
        funcname_a2h = "accelpy_test_a2hcopy"

        out.append(t_h2a.format(funcname=funcname_in, bound=bound_in,
                    varname=input["curname"], dtype=dtype_in))
        out.append(t_h2a.format(funcname=funcname_out, bound=bound_out,
                    varname=output["curname"], dtype=dtype_out))
        out.append(t_a2hcopy.format(funcname=funcname_a2h,
                    dtype=dtype_out, bound=bound_out))

        out.append(t_testfunc.format(varin=input["curname"],
                    varout=output["curname"]))

        return "\n".join(out)

    def getname_h2amalloc(self, arg):
        return "accelpy_h2amalloc_%s" % arg["curname"]

    def getname_h2acopy(self, arg):
        return "accelpy_h2acopy_%s" % arg["curname"]

    def getname_a2hcopy(self, arg):
        return "accelpy_a2hcopy_%s" % arg["curname"]


AccelBase.avails[FortranAccel.name] = FortranAccel
