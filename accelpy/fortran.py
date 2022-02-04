"""accelpy Fortran Accelerator module"""

from accelpy.kernel import KernelBase
from accelpy.util import fortline_pack
from ctypes import c_int32, c_int64, c_float, c_double


##########################
#  Code templates
##########################

t_module = """
MODULE accelpy_global
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

{datavars}

END MODULE
"""

t_main = """

{varmap}

INTEGER (C_INT64_T) FUNCTION accelpy_start() BIND(C, name="accelpy_start")

    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {usevarnames}
    IMPLICIT NONE

    accelpy_start = accelpy_kernel({usevarnames})

CONTAINS

{kernel}

END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_stop() BIND(C, name="accelpy_stop")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {usevarnames}
    IMPLICIT NONE

    accelpy_stop = 0

END FUNCTION
"""

t_kernel = """
INTEGER (C_INT64_T) FUNCTION accelpy_kernel({varnames})
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    {typedecls}

    {spec}

    accelpy_kernel = 0

END FUNCTION
"""

t_varmap = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data

    {varname} => data

    {funcname} = 0

END FUNCTION
"""

t_varmap_scalar = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION(1), INTENT(IN) :: data

    {varname} = data(1)

    {funcname} = 0

END FUNCTION
"""

class FortranKernel(KernelBase):

    name = "fortran"
    lang = "fortran"

    dtypemap = {
        "int32": ["INTEGER (C_INT32_T)", c_int32],
        "int64": ["INTEGER (C_INT64_T)", c_int64],
        "float32": ["REAL (C_FLOAT)", c_float],
        "float64": ["REAL (C_DOUBLE)", c_double]
    }

    def gen_code(self, compiler):

        macros = {}

        module_fmt = {
            "datavars": self._get_datavars(),
        }
        module = t_module.format(**module_fmt)

        main_fmt = {
            "varmap":self._gen_varmap(),
            "kernel":self._gen_kernel(),
            "usevarnames":self._gen_usevars(),
        }
        main = t_main.format(**main_fmt)

        #print(module)
        #print(main)
        #import pdb; pdb.set_trace()

        return [module, main], macros

    def getname_varmap(self, arg):
        return "accelpy_varmap_%s" % arg["curname"]

    def _get_datavars(self):

        out = []

        for arg in self.data:
            ndim = arg["data"].ndim
            dtype = self.get_dtype(arg)
            varname = arg["curname"]

            if ndim > 0:
                bound = ",".join([":"]*ndim)
                out.append("%s, DIMENSION(%s), POINTER :: %s" % (dtype, bound, varname))

            else:
                out.append("%s :: %s" % (dtype, varname))

        return "\n".join(out)

    def _gen_varmap(self):

        out = []

        for arg in self.data:
            dtype = self.get_dtype(arg)
            funcname = self.getname_varmap(arg)
            ndim = arg["data"].ndim

            if ndim > 0:
                bound = ",".join([str(s) for s in arg["data"].shape])
                out.append(t_varmap.format(funcname=funcname, varname=arg["curname"],
                            bound=bound, dtype=dtype))
            else:
                out.append(t_varmap_scalar.format(funcname=funcname,
                            varname=arg["curname"], dtype=dtype))

        return "\n".join(out)

    def _gen_kernel(self):

        _names = []
        typedecls = []

        for arg in self.data:
            ndim = arg["data"].ndim
            dtype = self.get_dtype(arg)
            varname = arg["curname"]

            _names.append(varname)

            if ndim > 0:
                if ("attrspec" in self.section.kwargs and
                    varname in self.section.kwargs["attrspec"] and
                    "dimension" in self.section.kwargs["attrspec"][varname]):
                    bound = self.section.kwargs["attrspec"][varname]["dimension"]

                else:
                    bound = ",".join([":"]*ndim)


                typedecls.append("%s, DIMENSION(%s), INTENT(INOUT) :: %s" % (
                            dtype, bound, varname))

            else:
                typedecls.append("%s, INTENT(IN) :: %s" % (dtype, varname))

        names = fortline_pack(_names)

        return t_kernel.format(spec="\n".join(self.section.body),
                typedecls="\n".join(typedecls), varnames="\n".join(names))

    def _gen_usevars(self):

        names = []

        for arg in self.data:

            names.append(arg["curname"])

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


FortranKernel.avails[FortranKernel.name] = FortranKernel
