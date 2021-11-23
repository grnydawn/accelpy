"""accelpy Fortran Accelerator module"""


from accelpy.accel import AccelBase, get_compilers
from ctypes import c_int32, c_int64, c_float, c_double

t_main = """

{testcode}

{datacopies}

INTEGER (C_INT64_T) FUNCTION accelpy_start(device, channel) BIND(C, name="accelpy_start")
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    INTEGER (C_INT64_T), INTENT(IN) :: device
    INTEGER (C_INT64_T), INTENT(IN) :: channel

    INTEGER (C_INT64_T) :: ACCELPY_WORKER_ID

    ACCELPY_WORKER_ID = 0
    
    accelpy_start = accelpy_kernel(device, channel, ACCELPY_WORKER_ID)

CONTAINS

{kernel}

END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_stop() BIND(C, name="accelpy_stop")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varattrs}
    IMPLICIT NONE

    {freemem}

    accelpy_stop = 0 
END FUNCTION
"""

t_module = """
MODULE accelpy_global
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

{typeattr}

{testvars}

{datavars}

CONTAINS

{procattr}

END MODULE
"""

t_kernel = """
INTEGER (C_INT64_T) FUNCTION accelpy_kernel(device, channel, ACCELPY_WORKER_ID)
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varandattr}
    IMPLICIT NONE

    INTEGER (C_INT64_T), INTENT(IN) :: device
    INTEGER (C_INT64_T), INTENT(IN) :: channel
    INTEGER (C_INT64_T), INTENT(IN) :: ACCELPY_WORKER_ID

    {order}

    accelpy_kernel = 0
END FUNCTION
"""

t_h2a = """
INTEGER (C_INT64_T) FUNCTION {funcname} (attrsize, attrs, data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}, {varname}_attr
    IMPLICIT NONE

    INTEGER (C_INT64_T), INTENT(IN) :: attrsize
    INTEGER (C_INT64_T), DIMENSION(attrsize), INTENT(IN) :: attrs
    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data

    {varname} => data
    ALLOCATE({varname}_attr%attrs(attrsize))
    {varname}_attr%attrs(:) = attrs(1:attrsize)

    {funcname} = 0

END FUNCTION
"""

t_a2hcopy = """
INTEGER (C_INT64_T) FUNCTION {funcname} (attrsize, attrs, data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}, {varname}_attr
    IMPLICIT NONE

    INTEGER (C_INT64_T), INTENT(IN) :: attrsize
    INTEGER (C_INT64_T), DIMENSION(attrsize), INTENT(IN) :: attrs
    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data

    {funcname} = 0

END FUNCTION
"""

t_testfunc = """
INTEGER (C_INT64_T) FUNCTION accelpy_test_run()  BIND(C, name="accelpy_test_run")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varin}, {varout}, {varin}_attr, {varout}_attr
    IMPLICIT NONE
    INTEGER :: id

    DO id=1, {varin}_attr%shape(1)
        {varout}(id) = {varin}(id)
    END DO

    DEALLOCATE({varin}_attr%attrs)
    DEALLOCATE({varout}_attr%attrs)

    accelpy_test_run = 0

END FUNCTION
"""

t_typeattr = """
TYPE :: accelpy_attrtype
    INTEGER (C_INT64_T), DIMENSION(:), ALLOCATABLE :: attrs

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
INTEGER (C_INT64_T) FUNCTION accelpy_ndim(self)
    CLASS(accelpy_attrtype), INTENT(IN) :: self

    accelpy_ndim = self%attrs(2)
END FUNCTION
INTEGER (C_INT64_T) FUNCTION accelpy_itemsize(self)
    CLASS(accelpy_attrtype), INTENT(IN) :: self

    accelpy_itemsize = self%attrs(3)
END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_size(self)
    CLASS(accelpy_attrtype), INTENT(IN) :: self

    accelpy_size = self%attrs(1)
END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_shape(self, dim)
    CLASS(accelpy_attrtype), INTENT(IN) :: self
    INTEGER, INTENT(IN) :: dim

    accelpy_shape = self%attrs(3+dim)
END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_stride(self, dim)
    CLASS(accelpy_attrtype), INTENT(IN) :: self
    INTEGER, INTENT(IN) :: dim

    accelpy_stride = self%attrs(3+self%attrs(2)+dim)
END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_unravel_index(self, tid, dim)
    CLASS(accelpy_attrtype), INTENT(IN) :: self
    INTEGER, INTENT(IN) :: dim, tid
    INTEGER :: i
    INTEGER (C_INT64_T) :: q, r, s

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
        "int32": ["INTEGER (C_INT32_T)", c_int32],
        "int64": ["INTEGER (C_INT64_T)", c_int64],
        "float32": ["REAL (C_FLOAT)", c_float],
        "float64": ["REAL (C_DOUBLE)", c_double]
    }

    def gen_code(self, inputs, outputs):

        module_fmt = {
            "typeattr": self._get_typeattr(),
            "testvars": self._get_testvars(),
            "datavars": self._get_datavars(inputs, outputs),
            "procattr": self._get_procattr(),
        }
        module = t_module.format(**module_fmt)

        main_fmt = {
            "testcode": self._get_testcode(),
            "datacopies":self._gen_datacopies(inputs, outputs),
            "kernel":self._gen_kernel(inputs, outputs),
            "varattrs":self._gen_varattrs(inputs, outputs),
            "freemem":self._gen_freemem(inputs, outputs)
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
            out.append("TYPE(accelpy_attrtype) :: %s_attr" % arg["curname"])

        return "\n".join(out)

    def _get_datavars(self, inputs, outputs):

        out = []

        for data in inputs+outputs:
            dtype = self.get_dtype(data)
            bound = ",".join([":"]*data["data"].ndim)
            out.append("%s, DIMENSION(%s), POINTER :: %s" % (dtype, bound, data["curname"]))
            out.append("TYPE(accelpy_attrtype) :: %s_attr" % data["curname"])

        return "\n".join(out)

    def _get_testcode(self):

        out = []

        input = self._testdata[0]
        output = self._testdata[1]

        dtype_in = self.get_dtype(input)
        dtype_out = self.get_dtype(input)
        funcname_in = "accelpy_test_h2acopy"
        funcname_out = "accelpy_test_h2amalloc"
        funcname_a2h = "accelpy_test_a2hcopy"
        bound_in = ",".join(["attrs(%d)"%(d+4) for d in range(input["data"].ndim)])
        bound_out = ",".join(["attrs(%d)"%(d+4) for d in range(output["data"].ndim)])

        out.append(t_h2a.format(funcname=funcname_in, bound=bound_in,
                    varname=input["curname"], dtype=dtype_in))
        out.append(t_h2a.format(funcname=funcname_out, bound=bound_out,
                    varname=output["curname"], dtype=dtype_out))
        out.append(t_a2hcopy.format(funcname=funcname_a2h, bound=bound_out,
                    varname=output["curname"], dtype=dtype_out))

        out.append(t_testfunc.format(varin=input["curname"],
                    varout=output["curname"]))

        return "\n".join(out)

    def _gen_datacopies(self, inputs, outputs):

        out = []

        for input in inputs:
            dtype = self.get_dtype(input)
            funcname = self.getname_h2acopy(input)
            bound = ",".join(["attrs(%d)"%(d+4) for d in range(input["data"].ndim)])

            out.append(t_h2a.format(funcname=funcname, varname=input["curname"],
                        bound=bound, dtype=dtype))

        for output in outputs:
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)
            bound = ",".join(["attrs(%d)"%(d+4) for d in range(output["data"].ndim)])

            out.append(t_h2a.format(funcname=funcname, varname=output["curname"],
                        bound=bound, dtype=dtype))

        for output in outputs:
            funcname = self.getname_a2hcopy(output)
            bound = ",".join(["attrs(%d)"%(d+4) for d in range(output["data"].ndim)])

            out.append(t_a2hcopy.format(funcname=funcname, varname=output["curname"],
                        bound=bound, dtype=dtype))

        return "\n".join(out)

    def _gen_kernel(self, inputs, outputs):

        names = []

        for data in inputs+outputs:
            names.append(data["curname"])
            names.append(data["curname"]+"_attr")

        order =  self._order.get_section(self.name)
        return t_kernel.format(order="\n".join(order.body), varandattr=", ".join(names))

    def _gen_freemem(self, inputs, outputs):

        out = []

        for data in inputs+outputs:
            out.append("DEALLOCATE(%s_attr%%attrs)" % data["curname"])

        return "\n".join(out)

    def _gen_varattrs(self, inputs, outputs):

        out = []

        for data in inputs+outputs:
            out.append("%s_attr" % data["curname"])

        return ", ".join(out)

    def getname_h2amalloc(self, arg):
        return "accelpy_h2amalloc_%s" % arg["curname"]

    def getname_h2acopy(self, arg):
        return "accelpy_h2acopy_%s" % arg["curname"]

    def getname_a2hcopy(self, arg):
        return "accelpy_a2hcopy_%s" % arg["curname"]


AccelBase.avails[FortranAccel.name] = FortranAccel
