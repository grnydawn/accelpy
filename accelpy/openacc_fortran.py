"""accelpy Fortran Accelerator module"""


from accelpy.accel import AccelBase, get_compilers
from accelpy.fortscan import get_firstexec
from ctypes import c_int32, c_int64, c_float, c_double

t_main = """

{testcode}

{datacopies}

INTEGER (C_INT64_T) FUNCTION accelpy_start() BIND(C, name="accelpy_start")

    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    accelpy_start = accelpy_kernel()

CONTAINS

{kernel}

END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_stop() BIND(C, name="accelpy_stop")
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

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

END MODULE
"""

t_kernel = """
INTEGER (C_INT64_T) FUNCTION accelpy_kernel()
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varandattr}
    IMPLICIT NONE

    {order}

    !$acc end parallel

    accelpy_kernel = 0
END FUNCTION
"""

t_h2acopy = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data

    {varname} => data
    !$acc enter data copyin({varname})

    {funcname} = 0

END FUNCTION
"""

t_h2amalloc = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data

    {varname} => data
    !$acc enter data create({varname})

    {funcname} = 0

END FUNCTION
"""

t_a2hcopy = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data

    !$acc exit data copyout({varname})

    {funcname} = 0

END FUNCTION
"""

t_testfunc = """
INTEGER (C_INT64_T) FUNCTION accelpy_test_run()  BIND(C, name="accelpy_test_run")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varin}, {varout}, {varin}_attr, {varout}_attr
    IMPLICIT NONE
    INTEGER :: id

    !$acc parallel num_gangs(1) num_workers({size}) vector_length(1)
    !$acc loop gang worker vector
    DO id=1, {varin}_attr%shape(1)
        {varout}(id) = {varin}(id)
    END DO
    !$acc end loop
    !$acc end parallel

    accelpy_test_run = 0

END FUNCTION
"""

t_typeattr = """
TYPE :: type_{varname}
    INTEGER :: ndim = {ndim}
    INTEGER :: size = {size}
    INTEGER, DIMENSION({ndim}) :: shape = (/ {shape} /)
    INTEGER, DIMENSION({ndim}) :: stride = (/ {stride} /)
END TYPE
"""


class OpenaccFortranAccel(AccelBase):

    name = "openacc_fortran"
    lang = "fortran"

    dtypemap = {
        "int32": ["INTEGER (C_INT32_T)", c_int32],
        "int64": ["INTEGER (C_INT64_T)", c_int64],
        "float32": ["REAL (C_FLOAT)", c_float],
        "float64": ["REAL (C_DOUBLE)", c_double]
    }

    def gen_code(self, compiler, inputs, outputs, triple, run_id, device, channel):

        macros = {
            "ACCELPY_OPENACC_RUNID": str(run_id),
            "ACCELPY_OPENACC_NGANGS": str(triple[0][0]*triple[0][1]*triple[0][2]),
            "ACCELPY_OPENACC_NWORKERS": str(triple[1][0]*triple[1][1]*triple[1][2]),
            "ACCELPY_OPENACC_LENVECTOR": str(triple[2][0]*triple[2][1]*triple[2][2]),
        }

        module_fmt = {
            "typeattr": self._get_typeattr(inputs, outputs),
            "testvars": self._get_testvars(),
            "datavars": self._get_datavars(inputs, outputs),
        }
        module = t_module.format(**module_fmt)

        main_fmt = {
            "testcode": self._get_testcode(),
            "datacopies":self._gen_datacopies(inputs, outputs),
            "kernel":self._gen_kernel(inputs, outputs)
        }
        main = t_main.format(**main_fmt)

        return [module, main], macros

    def _get_typeattr(self, inputs, outputs):

        out = []

        for arg in self._testdata+inputs+outputs:
            data = arg["data"]
            
            out.append(t_typeattr.format(varname=arg["curname"], ndim=str(data.ndim),
                        size=str(data.size), shape=self.get_shapestr(arg),
                        stride=self.get_stridestr(arg)))

        return "\n".join(out)

    def _get_testvars(self):

        out = []

        for arg in self._testdata:
            ndim = arg["data"].ndim
            dtype = self.get_dtype(arg)

            dimension = (("DIMENSION("+",".join([":"]*ndim)+"), POINTER")
                            if ndim > 0 else "")

            out.append("%s, %s :: %s" % (dtype, dimension, arg["curname"]))
            out.append("TYPE(type_{name}) :: {name}_attr".format(name=arg["curname"]))

        return "\n".join(out)

    def _get_datavars(self, inputs, outputs):

        out = []

        for data in inputs+outputs:
            dtype = self.get_dtype(data)
            bound = ",".join([":"]*data["data"].ndim)
            out.append("%s, DIMENSION(%s), POINTER :: %s" % (dtype, bound, data["curname"]))
            out.append("TYPE(type_{name}) :: {name}_attr".format(name=data["curname"]))

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
        bound_in = ",".join([str(s) for s in input["data"].shape])
        bound_out = ",".join([str(s) for s in output["data"].shape])

        out.append(t_h2acopy.format(funcname=funcname_in, bound=bound_in,
                    varname=input["curname"], dtype=dtype_in))
        out.append(t_h2amalloc.format(funcname=funcname_out, bound=bound_out,
                    varname=output["curname"], dtype=dtype_out))
        out.append(t_a2hcopy.format(funcname=funcname_a2h, bound=bound_out,
                    varname=output["curname"], dtype=dtype_out))

        out.append(t_testfunc.format(varin=input["curname"], size=str(output["data"].size),
                    varout=output["curname"]))

        return "\n".join(out)

    def _gen_datacopies(self, inputs, outputs):

        out = []

        for input in inputs:
            dtype = self.get_dtype(input)
            funcname = self.getname_h2acopy(input)
            bound = ",".join([str(s) for s in input["data"].shape])

            out.append(t_h2acopy.format(funcname=funcname, varname=input["curname"],
                        bound=bound, dtype=dtype))

        for output in outputs:
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)
            bound = ",".join([str(s) for s in output["data"].shape])

            out.append(t_h2amalloc.format(funcname=funcname, varname=output["curname"],
                        bound=bound, dtype=dtype))

        for output in outputs:
            funcname = self.getname_a2hcopy(output)
            bound = ",".join([str(s) for s in output["data"].shape])

            out.append(t_a2hcopy.format(funcname=funcname, varname=output["curname"],
                        bound=bound, dtype=dtype))

        return "\n".join(out)

    def _gen_kernel(self, inputs, outputs):

        names = []

        for data in inputs+outputs:
            names.append(data["curname"])
            names.append(data["curname"]+"_attr")

        order =  self._order.get_section(self.name)
        firstexec = get_firstexec(order.body)

        accpar = ["!$acc parallel num_gangs(ACCELPY_OPENACC_NGANGS) &",
                  "!$acc& num_workers(ACCELPY_OPENACC_NWORKERS) &",
                  "!$acc& vector_length(ACCELPY_OPENACC_LENVECTOR)"]

        body = order.body[:firstexec] + accpar + order.body[firstexec:]

        return t_kernel.format(order="\n".join(body), varandattr=", ".join(names))

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


OpenaccFortranAccel.avails[OpenaccFortranAccel.name] = OpenaccFortranAccel