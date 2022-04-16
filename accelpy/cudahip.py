"""accelpy CUDA and HIP Accelerator module"""

import os, sys

from collections import OrderedDict

from accelpy.util import get_c_dtype
from accelpy.accel import AccelBase


datasrc = """
#include <stdint.h>
#include <stdio.h>

{include}

{checkdef}

{moddvars}

extern "C" int64_t dataenter_{runid}({enterargs}) {{

    int64_t res;

    {entercopy}

    res = 0;

    return res;

}}

extern "C" int64_t dataexit_{runid}({exitargs}) {{

    int64_t res;

    {exitcopy}

    res = 0;

    return res;

}}
"""

kernelsrc = """
#include <stdint.h>
#include <stdio.h>

{include}

{checkdef}

{externs}

{macrodefs}

__global__ void device_kernel_{runid}({kernelargs}) {{

    {spec}

}}

extern "C" int64_t runkernel_{runid}({runkernelargs}) {{
    int64_t res;

    {kernelenter}

    {launchconf}

    device_kernel_{runid}<<<gridsize, blocksize>>>({launchargs});

    {kernelexit}

    res = 0;

    return res;
}}
"""


class CudaHipAccelBase(AccelBase):

    lang = "cpp"

    @classmethod
    def _gen_macrodefs(cls, localvars, modvars):

        typedefs = []
        consts = []
        macros = []

        macros.append("#define TYPE(varname) apy_type_##varname")
        macros.append("#define SHAPE(varname, dim) apy_shape_##varname##dim")
        macros.append("#define SIZE(varname) apy_size_##varname")

        for mhname, arg in modvars:
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            consts.append("const int64_t apy_size_%s = %d;" % (name, arg["data"].size))
            typedefs.append("typedef %s apy_type_%s;" % (dtype, name))

            if ndim > 0:

                shapestr = "".join(["[%d]"%s for s in arg["data"].shape])
                for d, s in enumerate(arg["data"].shape):
                    consts.append("const int64_t apy_shape_%s%d = %d;" % (name, d, s))
            else:
                pass

        for arg in localvars:
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            consts.append("const int64_t apy_size_%s = %d;" % (name, arg["data"].size))
            typedefs.append("typedef %s apy_type_%s;" % (dtype, name))

            if ndim > 0:

                shapestr = "".join(["[%d]"%s for s in arg["data"].shape])
                for d, s in enumerate(arg["data"].shape):
                    consts.append("const int64_t apy_shape_%s%d = %d;" % (name, d, s))
            else:
                pass

        return "\n".join(macros) + "\n\n" + "\n".join(typedefs) + "\n\n" + "\n".join(consts)

    @classmethod
    def _gen_launchconf(cls, secattr):

        grid, block = "dim3 gridsize(1);", "dim3 blocksize(1);"

        if "gridsize" in secattr:
            _grid = secattr["gridsize"]

            if isinstance(_grid , str):
                grid = _grid

            elif isinstance(_grid, int):
                grid = "dim3 gridsize(%d);" % _grid

            else:
                grid= "dim3 gridsize(%s);" % ", ".join([str(i) for i in _grid])

        if "blocksize" in secattr:
            _block = secattr["blocksize"]

            if isinstance(_block , str):
                block = _block

            elif isinstance(_block, int):
                block = "dim3 blocksize(%d);" % _block

            else:
                block= "dim3 blocksize(%s);" % ", ".join([str(i) for i in _block])

        return "%s\n%s\n" % (grid, block)

    @classmethod
    def gen_kernelfile(cls, knlhash, dmodname, runid, specid, modattr, section, workdir, localvars, modvars):

        kernelpath = os.path.join(workdir, "K%s%s" % (knlhash[2:], cls.srcext))

        attrspec = section.kwargs.get("attrspec", {})
        modattr.update(attrspec)

        externs = []
        runkernelargs = []
        kernelargs = []
        shapes = []
        launchargs = []
        kernelenter = []
        kernelexit = []

        for mhname, arg in modvars:
            mdname = "d" + mhname


            if arg["data"].ndim > 0:
                dtype = get_c_dtype(arg)
                hname = arg["curname"]
                shape = "".join(["[%d]"%s for s in arg["data"].shape])
                externs.append("extern %s (* %s)%s;" % (dtype, mdname, shape))
                runkernelargs.append("void * %s" % hname)
                launchargs.append("*%s" % mdname)
                kernelargs.append("%s %s%s" % (dtype, hname, shape))

                if arg["updateto"]:
                    kernelenter.append(cls._updateto(hname, mdname, arg["data"].size, dtype))

                if arg["updatefrom"]:
                    kernelexit.append(cls._updatefrom(hname, mdname, arg["data"].size, dtype))

        for arg in localvars:
            dtype = get_c_dtype(arg)
            hname = arg["curname"]
            dname = "d" + hname

            runkernelargs.append("void * %s" % hname)

            if arg["data"].ndim > 0:
                shape = "".join(["[%d]"%s for s in arg["data"].shape])
                launchargs.append("*%s" % dname)
                kernelargs.append("%s %s%s" % (dtype, hname, shape))

                kernelenter.append("//hipPointerAttribute_t attr;")
                kernelenter.append("//hipPointerGetAttributes(&attr, %s);" % hname)
                kernelenter.append("//printf(\"dptr, %s : %%p\\n\", %s);" % (mname, mname))
                kernelenter.append("//printf(\"attr : %p\\n\", attr.devicePointer);")

            else:
                launchargs.append("*((%s *) %s)" % (dtype, hname))
                kernelargs.append("%s %s" % (dtype, hname))

        kernelenter.append(cls._gen_enterfini())
        kernelexit.append(cls._gen_exitfini())

        #kernelenter.append("printf(\"KENTER\\n\");")
        #kernelexit.append("printf(\"KEXIT\\n\");")

        kernelparams = {
            "runid": str(runid) + str(specid),
            "externs": "\n".join(externs),
            "include": "\n".join(cls._gen_include()),
            "runkernelargs": ", ".join(runkernelargs),
            "kernelargs": ", ".join(kernelargs),
            "launchargs": ", ".join(launchargs),
            "kernelenter": "\n".join(kernelenter),
            "kernelexit": "\n".join(kernelexit),
            "launchconf": cls._gen_launchconf(section.kwargs),
            "macrodefs": cls._gen_macrodefs(localvars, modvars),
            "checkdef" : cls._gen_checkdef(),
            "spec": "\n".join(section.body),
        }

        with open(kernelpath, "w") as fkernel:
            fkernel.write(kernelsrc.format(**kernelparams))

        #import pdb; pdb.set_trace()
        return kernelpath

    @classmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
                        copyin, copyout, alloc, attr):

        datapath = os.path.join(workdir, filename)

        dataparams = {"runid": str(runid), "datamodname": modname,
                      "include": "\n".join(cls._gen_include())}

        moddvars = []
        enterargs = []
        exitargs = []
        entercopy = []
        exitcopy = []

        for cio in copyinout:
            hname = cio["curname"]
            dname = "d" + hname
            dtype = get_c_dtype(cio)

            enterargs.append("void * " + hname)
            exitargs.append("void * " + hname)

            if cio["data"].ndim > 0:
                shape = "".join(["[%d]"%s for s in cio["data"].shape])
                moddvars.append("%s (* %s)%s;" % (dtype, dname, shape))
                entercopy.append(cls._mapto(hname, dname, cio["data"].size, dtype))
                exitcopy.append(cls._mapfrom(hname, dname, cio["data"].size, dtype))

        for ci in copyin:
            hname = ci["curname"]
            dname = "d" + hname
            dtype = get_c_dtype(ci)

            enterargs.append("void * " + hname)

            if ci["data"].ndim > 0:
                shape = "".join(["[%d]"%s for s in ci["data"].shape])
                moddvars.append("%s (* %s)%s;" % (dtype, dname, shape))
                entercopy.append(cls._mapto(hname, dname, ci["data"].size, dtype))
                exitcopy.append(cls._mapdelete(dname))

        for co in copyout:
            hname = co["curname"]
            dname = "d" + hname
            dtype = get_c_dtype(co)

            enterargs.append("void * " + hname)
            exitargs.append("void * " + hname)

            if co["data"].ndim > 0:
                shape = "".join(["[%d]"%s for s in co["data"].shape])
                moddvars.append("%s (* %s)%s;" % (dtype, dname, shape))
                entercopy.append(cls._mapalloc(dname, co["data"].size, dtype))
                exitcopy.append(cls._mapfrom(hname, dname, co["data"].size, dtype))

        for al in alloc:
            hname = al["curname"]
            dname = "d" + hname
            dtype = get_c_dtype(al)

            enterargs.append("void * " + hname)

            if al["data"].ndim > 0:
                shape = "".join(["[%d]"%s for s in al["data"].shape])
                moddvars.append("%s (* %s)%s;" % (dtype, dname, shape))
                entercopy.append(cls._mapalloc(dname, al["data"].size, dtype))
                exitcopy.append(cls._mapdelete(dname))

        entercopy.append(cls._gen_enterfini())
        exitcopy.append(cls._gen_exitfini())

        dataparams["moddvars"]  = "\n".join(moddvars)
        dataparams["entercopy"] = "\n".join(entercopy)
        dataparams["exitcopy"]  = "\n".join(exitcopy)
        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["exitargs"]  = ", ".join(exitargs)
        dataparams["checkdef"]  = cls._gen_checkdef() 

        with open(datapath, "w") as fdata:
            fdata.write(datasrc.format(**dataparams))

        #import pdb; pdb.set_trace()
        return datapath


class CudaAccel(CudaHipAccelBase):
    accel = "cuda"
    srcext = ".cu"

    @classmethod
    def _updateto(cls, hname, dname, size, tname):

        fmt = ("cudaMemcpy({dname}, {hname}, {size} * sizeof({type}), cudaMemcpyHostToDevice);\n"
               "CHECK_API();\n")

        return fmt.format(hname=hname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _updatefrom(cls, hname, dname, size, tname):

        fmt = ("cudaMemcpy({hname}, {dname}, {size} * sizeof({type}), cudaMemcpyDeviceToHost);\n"
               "CHECK_API();\n")

        return fmt.format(hname=hname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapto(cls, hname, dname, size, tname):

        return (cls._mapalloc(dname, size, tname) +
                cls._updateto(hname, dname, size, tname))

    @classmethod
    def _mapfrom(cls, hname, dname, size, tname):

        return (cls._updatefrom(hname, dname, size, tname) +
                cls._mapdelete(dname))

    @classmethod
    def _mapalloc(cls, dname, size, tname):

        fmt = ("cudaMalloc((void **)&{dname}, {size} * sizeof({type}));\n"
               "CHECK_API();")

        return fmt.format(dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapdelete(cls, dname):

        return "cudaFree(%s);\nCHECK_API();"  % dname

    @classmethod
    def _gen_include(cls):

        return ["#include <cuda_runtime_api.h>", "#include <cuda.h>"]

    @classmethod
    def _gen_checkdef(cls):
        return """
void CHECK_API(void)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error: %s\\n", cudaGetErrorString(err));
        exit(err);
    }
}
"""

    @classmethod
    def _gen_enterfini(cls):
        return "cudaDeviceSynchronize();\nCHECK_API();"

    @classmethod
    def _gen_exitfini(cls):
        return "cudaDeviceSynchronize();\nCHECK_API();"


class HipAccel(CudaHipAccelBase):
    accel = "hip"
    srcext = ".cpp"

    @classmethod
    def _updateto(cls, hname, dname, size, tname):

        fmt = ("hipMemcpyHtoD({dname}, {hname}, {size} * sizeof({type}));\n"
               "CHECK_API();\n")

        return fmt.format(hname=hname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _updatefrom(cls, hname, dname, size, tname):

        fmt = ("hipMemcpyDtoH({hname}, {dname}, {size} * sizeof({type}));\n"
               "CHECK_API();\n")

        return fmt.format(hname=hname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapto(cls, hname, dname, size, tname):

        return (cls._mapalloc(dname, size, tname) +
                cls._updateto(hname, dname, size, tname))

    @classmethod
    def _mapfrom(cls, hname, dname, size, tname):

        return (cls._updatefrom(hname, dname, size, tname) +
                cls._mapdelete(dname))

    @classmethod
    def _mapalloc(cls, dname, size, tname):

        fmt = ("hipMalloc((void **)&{dname}, {size} * sizeof({type}));\n"
               "CHECK_API();\n")

        return fmt.format(dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapdelete(cls, dname):

        return "hipFree(%s);\nCHECK_API();\n"  % dname

    @classmethod
    def _gen_include(cls):

        return ["#include <hip/hip_runtime.h>"]

    @classmethod
    def _gen_checkdef(cls):
        return """
void CHECK_API(void)
{
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        printf("Error: %s\\n", hipGetErrorString(err));
        exit(err);
    }
}
"""

    @classmethod
    def _gen_enterfini(cls):
        return "hipDeviceSynchronize();\nCHECK_API();"

    @classmethod
    def _gen_exitfini(cls):
        return "hipDeviceSynchronize();\nCHECK_API();"

