"""accelpy Fortran Accelerator module"""


from accelpy.accel import AccelBase, get_compilers
from accelpy.util import fortline_pack
from ctypes import c_int32, c_int64, c_float, c_double

t_main = """

{testcode}

{datacopies}

INTEGER (C_INT64_T) FUNCTION accelpy_start() BIND(C, name="accelpy_start")

    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varnames}
    IMPLICIT NONE

    accelpy_start = accelpy_kernel({varnames})

CONTAINS

{kernel}

END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_stop() BIND(C, name="accelpy_stop")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varnames}
    IMPLICIT NONE

    {nullify}

    accelpy_stop = 0 

END FUNCTION
"""

t_module = """
MODULE accelpy_global
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

{testvars}

{datavars}

END MODULE
"""

t_kernel = """
INTEGER (C_INT64_T) FUNCTION accelpy_kernel({varnames})
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    {typedecls}

    {order}

    accelpy_kernel = 0

END FUNCTION
"""

t_h2a = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data

    {varname} => data

    {funcname} = 0

END FUNCTION
"""

t_h2a_scalar = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION(1), INTENT(IN) :: data

    {varname} = data(1)

    {funcname} = 0

END FUNCTION
"""

t_a2hcopy = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(OUT) :: data

    {funcname} = 0

END FUNCTION
"""

t_a2hcopy_scalar = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION(1), INTENT(OUT) :: data

    {funcname} = 0

END FUNCTION
"""

t_testfunc = """
INTEGER (C_INT64_T) FUNCTION accelpy_test_run()  BIND(C, name="accelpy_test_run")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varin}, {varout}
    IMPLICIT NONE
    INTEGER :: id

    DO id=LBOUND({varin},1), UBOUND({varin},1)
        {varout}(id) = {varin}(id)
    END DO

    IF (ASSOCIATED({varin})) THEN
        NULLIFY({varin})
    END IF

    IF (ASSOCIATED({varout})) THEN
        NULLIFY({varout})
    END IF

    accelpy_test_run = 0

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

    def gen_code(self, compiler, inputs, order, outputs,
                 worker_triple, run_id, device, channel):

        macros = {}
                
        module_fmt = {
            "testvars": self._get_testvars(),
            "datavars": self._get_datavars(inputs, outputs),
        }
        module = t_module.format(**module_fmt)

        main_fmt = {
            "testcode": self._get_testcode(),
            "datacopies":self._gen_datacopies(inputs, outputs),
            "kernel":self._gen_kernel(inputs, order, outputs),
            "varnames":self._gen_usevars(inputs, outputs),
            "nullify":self._gen_nullify(inputs, outputs)
        }
        main = t_main.format(**main_fmt)

        #print(module)
        #print(main)
        #import pdb; pdb.set_trace()
        return [module, main], macros

#    def _get_procattr(self, inputs, outputs):
#
#        out = []
#
#        for arg in self._testdata+inputs+outputs:
#            data = arg["data"]
#
#            if data.ndim > 0:
#                out.append(t_procattr.format(varname=arg["curname"]))
#
#        return "\n".join(out)
#
#    def _get_typeattr(self, inputs, outputs):
#
#        out = []
#
#        for arg in self._testdata+inputs+outputs:
#            data = arg["data"]
#
#            if data.ndim > 0:
#                out.append(t_typeattr.format(varname=arg["curname"],
#                        ndim=str(data.ndim), size=str(data.size),
#                        shape=self.get_shapestr(arg),
#                        stride=self.get_stridestr(arg)))
#
#        return "\n".join(out)

    def _get_testvars(self):

        out = []

        for arg in self._testdata:
            ndim = arg["data"].ndim
            dtype = self.get_dtype(arg)

            attrspec = (("DIMENSION("+",".join([":"]*ndim)+"), POINTER")
                            if ndim > 0 else "")

            out.append("%s, %s :: %s" % (dtype, attrspec, arg["curname"]))

        return "\n".join(out)

    def _get_datavars(self, inputs, outputs):

        out = []

        for data in inputs+outputs:
            ndim = data["data"].ndim
            dtype = self.get_dtype(data)
            varname = data["curname"]

            if ndim > 0:
                bound = ",".join([":"]*ndim)
                out.append("%s, DIMENSION(%s), POINTER :: %s" % (dtype, bound, varname))

            else:
                out.append("%s :: %s" % (dtype, varname))

        return "\n".join(out)

    def _get_testcode(self):

        out = []

        input = self._testdata[0]
        output = self._testdata[1]

        dtype_in = self.get_dtype(input)
        dtype_out = self.get_dtype(output)
        funcname_in = "accelpy_test_h2acopy"
        funcname_out = "accelpy_test_h2amalloc"
        funcname_a2h = "accelpy_test_a2hcopy"
        bound_in = ",".join([str(s) for s in input["data"].shape])
        bound_out = ",".join([str(s) for s in output["data"].shape])

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
            ndim = input["data"].ndim

            if ndim > 0:
                bound = ",".join([str(s) for s in input["data"].shape])
                out.append(t_h2a.format(funcname=funcname, varname=input["curname"],
                            bound=bound, dtype=dtype))
            else:
                out.append(t_h2a_scalar.format(funcname=funcname,
                            varname=input["curname"], dtype=dtype))

        for output in outputs:
            dtype = self.get_dtype(output)
            funcname = self.getname_h2amalloc(output)
            ndim = output["data"].ndim

            if ndim > 0:
                bound = ",".join([str(s) for s in output["data"].shape])
                out.append(t_h2a.format(funcname=funcname, varname=output["curname"],
                        bound=bound, dtype=dtype))

            else:
                out.append(t_h2a_scalar.format(funcname=funcname,
                        varname=output["curname"], dtype=dtype))

        for output in outputs:
            dtype = self.get_dtype(output)
            funcname = self.getname_a2hcopy(output)
            ndim = output["data"].ndim

            if ndim > 0:
                bound = ",".join([str(s) for s in output["data"].shape])
                out.append(t_a2hcopy.format(funcname=funcname, varname=output["curname"],
                        bound=bound, dtype=dtype))

            else:
                out.append(t_a2hcopy_scalar.format(funcname=funcname,
                        varname=output["curname"], dtype=dtype))

        return "\n".join(out)

    def _gen_usevars(self, inputs, outputs):

        names = []

        for data in inputs+outputs:

            if data["data"].ndim > 0:
                names.append(data["curname"])

        lines = [""]
        maxlen = 72

        for name in names:
            if len(lines[-1]) + len(name) > maxlen:            

                lines[-1] += " &"
                lines.append("        &, %s" % name)

            elif lines[-1] == "":
                lines[-1] += name

            else:
                lines[-1] += ", " + name

        return "\n".join(lines)

    def _gen_nullify(self, inputs, outputs):

        lines = []

        for data in inputs+outputs:
            if data["data"].ndim > 0:
                lines.append("IF (ASSOCIATED(%s)) THEN" % data["curname"])
                lines.append("        NULLIFY(%s)" % data["curname"])
                lines.append("    END IF")

        return "\n".join(lines)

    def _gen_kernel(self, inputs, order, outputs):

        _names = []
        typedecls = []

        for data in inputs+outputs:
            ndim = data["data"].ndim
            dtype = self.get_dtype(data)
            varname = data["curname"]

            _names.append(varname)

            if ndim > 0:
                if ("attrspec" in order.kwargs and
                    varname in order.kwargs["attrspec"] and
                    "dimension" in order.kwargs["attrspec"][varname]):
                    bound = order.kwargs["attrspec"][varname]["dimension"]

                else:
                    bound = ",".join([":"]*ndim)

                typedecls.append("%s, DIMENSION(%s), INTENT(INOUT) :: %s" % (
                            dtype, bound, varname))

            else:
                typedecls.append("%s, INTENT(IN) :: %s" % (dtype, varname))


        names = fortline_pack(_names)

        return t_kernel.format(order="\n".join(order.body),
                typedecls="\n".join(typedecls), varnames="\n".join(names))

#    def _gen_varattrs(self, inputs, outputs):
#
#        out = []
#
#        for data in inputs+outputs:
#            out.append("%s_attr" % data["curname"])
#
#        return ", ".join(out)

    def getname_h2amalloc(self, arg):
        return "accelpy_h2amalloc_%s" % arg["curname"]

    def getname_h2acopy(self, arg):
        return "accelpy_h2acopy_%s" % arg["curname"]

    def getname_a2hcopy(self, arg):
        return "accelpy_a2hcopy_%s" % arg["curname"]


FortranAccel.avails[FortranAccel.name] = FortranAccel
