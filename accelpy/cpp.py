"""accelpy C++ kernel module"""

from accelpy.kernel import KernelBase
from ctypes import c_int32, c_int64, c_float, c_double

##########################
#  Code templates
##########################

t_main = """
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>


{varmap}

{kernel}

extern "C" int64_t accelpy_start() {{

    int64_t res;

    {argdefs}

    res = accelpy_kernel({startargs});

    return res;
}}

extern "C" int64_t accelpy_stop() {{

    int64_t res;

    res = 0;

    return res;
}}
"""

t_kernel = """
extern "C" int64_t accelpy_kernel({kernelargs}){{

    int64_t res;

    {shape}

    {spec}

    res = 0;

    return res;
}}
"""

t_varmap = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;

    {varname} = ({dtype} *) data;

    res = 0;

    return res;
}}
"""

t_varmap_scalar = """
extern "C" int64_t {funcname}(void * data) {{
    int64_t res;

    {varname} = *({dtype} *) data;

    res = 0;

    return res;
}}
"""


class CppKernel(KernelBase):

    name = "cpp"
    lang = "cpp"

    # dtype: ( C type name, ctype )
    # TODO: dynamically generates on loading
    dtypemap = {
        "int32": ["int32_t", c_int32],
        "int64": ["int64_t", c_int64],
        "float32": ["float", c_float],
        "float64": ["double", c_double]
    }

    def gen_code(self, compiler):

        macros = {}

        argdefs, startargs = self._gen_recast()

        main_fmt = {
            "kernel":self._gen_kernel(),
            "varmap":self._gen_varmap(),
            "argdefs":argdefs,
            "startargs":startargs,
        }
        main = t_main.format(**main_fmt)

        #print(module)
        #print(main)
        #import pdb; pdb.set_trace()

        return [main], macros

    def getname_varmap(self, arg):
        return "accelpy_varmap_%s" % arg["curname"]

    def _gen_varmap(self):

        out = []

        for arg in self.data:
            dtype = self.get_dtype(arg)
            funcname = self.getname_varmap(arg)
            varname = "accelpy_var_" + arg["curname"]
            ndim = arg["data"].ndim

            if ndim > 0:
                out.append("%s * %s;" % (dtype, varname))
                out.append(t_varmap.format(funcname=funcname, varname=varname,
                            dtype=dtype))
            else:
                out.append("%s %s;" % (dtype, varname))
                out.append(t_varmap_scalar.format(funcname=funcname,
                            varname=varname, dtype=dtype))

        return "\n".join(out)

    def _gen_kernel(self):

        args = []
        shapes = []

        for arg in self.data:
            ndim = arg["data"].ndim
            dtype = self.get_dtype(arg)
            name = arg["curname"]

            if ndim > 0:
                shape0 = ",".join(["[%d]"%s for s in arg["data"].shape])
                shape1 = ",".join([str(s) for s in arg["data"].shape])

                shapes.append("int shape_%s[%d] = {%s};" % (name, ndim, shape1))
                args.append("%s %s%s" % (dtype, name, shape0))

            else:
                args.append("%s %s" % (dtype, name))

        return t_kernel.format(spec="\n".join(self.section.body),
                kernelargs=", ".join(args), shape="\n".join(shapes))

    def _gen_recast(self):

        argdefs = []
        startargs = []

        fmt = "{dtype}(*ptr_{name}){shape} = reinterpret_cast<{dtype}(*){shape}>(accelpy_var_{name});"

        for arg in self.data:
            dtype = self.get_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            if ndim > 0:
                shape = ",".join(["[%d]"%s for s in arg["data"].shape])
                argdefs.append(fmt.format(dtype=dtype, name=name, shape=shape))
                startargs.append("(*ptr_" + name + ")")

            else:
                startargs.append("accelpy_var_" + name)

        return "\n".join(argdefs), ", ".join(startargs)


class OpenmpCppKernel(CppKernel):
    name = "openmp_cpp"


class OpenaccCppKernel(CppKernel):
    name = "openacc_cpp"


OpenmpCppKernel.avails[OpenmpCppKernel.name] = OpenmpCppKernel
OpenaccCppKernel.avails[OpenaccCppKernel.name] = OpenaccCppKernel
CppKernel.avails[CppKernel.name] = CppKernel
