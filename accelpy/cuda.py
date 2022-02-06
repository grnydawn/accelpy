"""accelpy CUDA kernel module"""

from accelpy.cpp import CppKernel


class CudaKernel(CppKernel):

    name = "cuda"
    lang = "cpp"

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


CudaKernel.avails[CudaKernel.name] = CudaKernel
