"""accelpy HIP kernel module"""

from accelpy.cpp import CppKernel


class HipKernel(CppKernel):

    name = "hip"
    lang = "cpp"

    def get_include(self):
        return "#include <hip/hip_runtime.h>"

    def gen_kernel(self):

        kernel_sec = self.spec.get_section(self.section.kwargs["kernel"])

        return "\n".join(kernel_sec.body)

    def gen_startmain(self):

        argdefs = []

        fmt = "{dtype}(*apy_ptr_{name}){shape} = reinterpret_cast<{dtype}(*){shape}>(accelpy_var_{name});"

        for arg in self.data:
            dtype = self.get_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            if ndim > 0:
                shape = "".join(["[%d]"%s for s in arg["data"].shape])
                argdefs.append(fmt.format(dtype=dtype, name=name, shape=shape))
                argdefs.append("%s(*apy_dptr_%s)%s;" % (dtype, name, shape))

        return "\n".join(argdefs) + "\n\n" + "\n".join(self.section.body)

    def gen_preprop(self):

        typedefs = []
        consts = []
        macros = []

        # TYPE(x), SHAPE(x, 0), SIZE(x), ARG(x), DVAR(x), FLATTEN(x)

        macros.append("#define TYPE(varname) apy_type_##varname")
        macros.append("#define SHAPE(varname, dim) apy_shape_##varname##dim")
        macros.append("#define SIZE(varname) apy_size_##varname")
        macros.append("#define ARG(varname) apy_type_##varname varname apy_shapestr_##varname")
        macros.append("#define VAR(varname) (*apy_ptr_##varname)")
        macros.append("#define DVAR(varname) (*apy_dptr_##varname)")
        macros.append("#define PTR(varname) apy_ptr_##varname")
        macros.append("#define DPTR(varname) apy_dptr_##varname")
        macros.append("#define FLATTEN(varname) accelpy_var_##varname")

        for arg in self.data:
            dtype = self.get_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            consts.append("__host__ __device__ const int apy_size_%s = %d;" % (name, arg["data"].size))
            typedefs.append("typedef %s apy_type_%s;" % (dtype, name))

            if ndim > 0:

                shapestr = "".join(["[%d]"%s for s in arg["data"].shape])
                macros.append("#define apy_shapestr_%s %s" % (name, shapestr))
                for d, s in enumerate(arg["data"].shape):
                    consts.append("__host__ __device__ const int apy_shape_%s%d = %d;" % (name, d, s))
            else:
                pass

        return "\n".join(macros) + "\n\n" + "\n".join(typedefs) + "\n\n" + "\n".join(consts)

HipKernel.avails[HipKernel.name] = HipKernel
