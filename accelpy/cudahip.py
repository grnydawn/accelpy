"""accelpy CUDA and HIP Accelerator module"""

import os, sys

from collections import OrderedDict

from accelpy.util import get_c_dtype
from accelpy.accel import AccelBase


datasrc = """
#include <stdint.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

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
#include <hip/hip_runtime.h>

{externs}

{macrodefs}

__global__ void device_kernel_{runid}({kernelargs}) {{

    {spec}

}}

extern "C" int64_t runkernel_{runid}({runkernelargs}) {{
    int64_t res;

    hipPointerAttribute_t attr;

    {kernelenter}

    device_kernel_{runid}<<<{launchconf}>>>({launchargs});

    {kernelexit}

    return res;
}}
"""


class CudaHipAccelBase(AccelBase):

    lang = "cpp"
    srcext = ".cpp"

    def _mapto(cls, vname, dname, size, tname):
        raise NotImplementedError("_mapto")

    def _mapfrom(cls, vname, dname, size, tname):
        raise NotImplementedError("_mapfrom")

    def _mapalloc(cls, dname, size, tname):
        raise NotImplementedError("_mapalloc")

    def _mapdelete(cls, dname):
        raise NotImplementedError("_mapdelete")

    @classmethod
    def _gen_macrodefs(cls, localvars, modvars):

        typedefs = []
        consts = []
        macros = []

        macros.append("#define TYPE(varname) apy_type_##varname")
        macros.append("#define SHAPE(varname, dim) apy_shape_##varname##dim")
        macros.append("#define SIZE(varname) apy_size_##varname")

        for oldname, arg in modvars:
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

        if "launch" in secattr:
            _conf = secattr["launch"]

            if isinstance(_conf, str):
                return _conf

            elif isinstance(_conf, int):
                return "1, %d" % _conf

            elif len(_conf) == 2:

                conf0 = _conf[0]

                if isinstance(_conf[0], int):
                    conf0 = str(_conf[0])

                elif isinstance(_conf[0], (tuple, list)): 
                    conf0= "dim3(%s)" % ", ".join([str(i) for i in _conf[0]])

                conf1 = _conf[1]

                if isinstance(_conf[1], int):
                    conf1 = str(_conf[1])

                elif isinstance(_conf[1], (tuple, list)): 
                    conf1= "dim3(%s)" % ", ".join([str(i) for i in _conf[1]])

                return "%s, %s" % (conf0, conf1)

        else:
            return "1, 1"

    @classmethod
    def gen_kernelfile(cls, knlhash, dmodname, runid, section, workdir, localvars, modvars):

        kernelpath = os.path.join(workdir, "K%s%s" % (knlhash[2:], cls.srcext))

        externs = []
        runkernelargs = []
        kernelargs = []
        shapes = []
        launchargs = []
        kernelenter = []
        kernelexit = []

        for mname, arg in modvars:
            if arg["data"].ndim > 0:
                dtype = get_c_dtype(arg)
                hname = arg["curname"]
                shape = "".join(["[%d]"%s for s in arg["data"].shape])

                externs.append("%s (* %s)%s;" % (dtype, mname, shape))
                launchargs.append("*%s" % mname)
                kernelargs.append("%s %s%s" % (dtype, hname, shape))

        for arg in localvars:
            dtype = get_c_dtype(arg)
            hname = arg["curname"]
            dname = "d" + name

            runkernelargs.append("void * %s" % hname)

            if arg["data"].ndim > 0:
                shape = "".join(["[%d]"%s for s in arg["data"].shape])
                launchargs.append("*%s" % dname)
                kernelargs.append("%s %s%s" % (dtype, hname, shape))
                kernelenter.append("//hipPointerGetAttributes(&attr, %s);" % hname)
                kernelenter.append("//printf(\"dptr, %s : %%p\\n\", %s);" % (mname, mname))
                kernelenter.append("//printf(\"attr : %p\\n\", attr.devicePointer);")

            else:
                launchargs.append("*((%s *) %s)" % (dtype, hname))
                kernelargs.append("%s %s" % (dtype, hname))

        kernelparams = {
            "runid": str(runid),
            "externs": "\n".join(externs),
            "runkernelargs": ", ".join(runkernelargs),
            "kernelargs": ", ".join(kernelargs),
            "launchargs": ", ".join(launchargs),
            "kernelenter": "\n".join(kernelenter),
            "kernelexit": "\n".join(kernelexit),
            "launchconf": cls._gen_launchconf(section.kwargs),
            "macrodefs": cls._gen_macrodefs(localvars, modvars),
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

        dataparams = {"runid": str(runid), "datamodname": modname}

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

        dataparams["moddvars"]  = "\n".join(moddvars)
        dataparams["entercopy"] = "\n".join(entercopy)
        dataparams["exitcopy"]  = "\n".join(exitcopy)
        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["exitargs"]  = ", ".join(exitargs)

        with open(datapath, "w") as fdata:
            fdata.write(datasrc.format(**dataparams))

        #import pdb; pdb.set_trace()
        return datapath


class CudaAccel(CudaHipAccelBase):
    accel = "cuda"

    @classmethod
    def _mapto(cls, hname, dname, size, tname):

        fmt = ("cudaMalloc((void **)&{dname}, {size} * sizeof({type}));\n"
               "cudaMemcpy({dname}, {hname}, {size} * sizeof({type}), cudaMemcpyHostToDevice);\n")

        return fmt.format(hname=hname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapfrom(cls, hname, dname, size, tname):

        fmt = ("cudaMemcpy({hname}, {dname}, {size} * sizeof({type}), cudaMemcpyDeviceToHost);\n"
               "cudaFree({dname});\n")

        return fmt.format(hname=hname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapalloc(cls, dname, size, tname):

        fmt = "cudaMalloc((void **)&{dname}, {size} * sizeof({type}));\n"

        return fmt.format(dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapdelete(cls, dname):

        return "cudaFree(%s);\n" % dname


class HipAccel(CudaHipAccelBase):
    accel = "hip"

    @classmethod
    def _mapto(cls, hname, dname, size, tname):

        fmt = ("hipMalloc((void **)&{dname}, {size} * sizeof({type}));\n"
               "hipMemcpyHtoD({dname}, {hname}, {size} * sizeof({type}));\n")

        return fmt.format(hname=hname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapfrom(cls, hname, dname, size, tname):

        fmt = ("hipMemcpyDtoH({hname}, {dname}, {size} * sizeof({type}));\n"
               "hipFree({dname});\n")

        return fmt.format(hname=hname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapalloc(cls, dname, size, tname):

        fmt = "hipMalloc((void **)&{dname}, {size} * sizeof({type}));\n"

        return fmt.format(dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapdelete(cls, dname):

        return "hipFree(%s);\n" % dname

_chaccels = OrderedDict()
AccelBase.avails[CudaHipAccelBase.lang] = _chaccels

_chaccels[CudaAccel.accel] = CudaAccel
_chaccels[HipAccel.accel] = HipAccel

