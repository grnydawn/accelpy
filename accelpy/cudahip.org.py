"""accelpy CUDA and HIP Accelerator module"""

import os, sys

from collections import OrderedDict

from accelpy.util import get_c_dtype
from accelpy.accel import AccelBase


convfmt = "{dtype}(*apy_ptr_{name}){shape} = reinterpret_cast<{dtype}(*){shape}>({orgname});"

datasrc = """
#include <stdint.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

{modvars}

extern "C" int64_t dataenter_{runid}({enterargs}) {{

    int64_t res;

    {enterassign}

    {enterapicall}

    res = 0;

    return res;

}}

extern "C" int64_t dataexit_{runid}({exitargs}) {{

    int64_t res;

    {exitapicall}

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

__global__ void device_kernel({devicekernelargs}) {{

    {reshape}

    {spec}

}}

int64_t kernel_{runid}({kernelargs}) {{
    int64_t res;

    {kernelenter}

    device_kernel<<<{launchconf}>>>({launchargs});

    {kernelexit}

    res = 0;

    return res;

}}

extern "C" int64_t runkernel_{runid}({runkernelargs}) {{
    int64_t res;

    {startmain}

    res = kernel_{runid}({actualargs});

    return res;
}}
"""


class CudaHipAccelBase(AccelBase):

    lang = "cpp"
    srcext = ".cpp"
    libext = ".dylib" if sys.platform == "darwin" else ".so"

    def _mapto(cls, vname, dname, size, tname):
        raise NotImplementedError("_mapto")

    def _mapfrom(cls, vname, dname, size, tname):
        raise NotImplementedError("_mapfrom")

    def _mapalloc(cls, dname, size, tname):
        raise NotImplementedError("_mapalloc")

    @classmethod
    def _gen_macrodefs(cls, localvars, modvars):

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

        for oldname, arg in modvars:
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            consts.append("const int64_t apy_size_%s = %d;" % (name, arg["data"].size))
            typedefs.append("typedef %s apy_type_%s;" % (dtype, name))

            if ndim > 0:

                shapestr = "".join(["[%d]"%s for s in arg["data"].shape])
                macros.append("#define apy_shapestr_%s %s" % (name, shapestr))
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
                macros.append("#define apy_shapestr_%s %s" % (name, shapestr))
                for d, s in enumerate(arg["data"].shape):
                    consts.append("const int64_t apy_shape_%s%d = %d;" % (name, d, s))
            else:
                pass

        return "\n".join(macros) + "\n\n" + "\n".join(typedefs) + "\n\n" + "\n".join(consts)

    @classmethod
    def _gen_kernelargs(cls, localvars, modvars):

        args = []
        dargs = []
        shapes = []
        externs = []

        for modname, arg in modvars:
            ndim = arg["data"].ndim
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            dname = "d" + name

            externs.append("extern void * %s;" % modname)
            dargs.append(dname)

            if ndim > 0:
                shape0 = "".join(["[%d]"%s for s in arg["data"].shape])
                shape1 = ",".join([str(s) for s in arg["data"].shape])

                shapes.append("const int64_t shape_%s[%d] = {%s};" % (name, ndim, shape1))
                args.append("%s %s%s" % (dtype, name, shape0))

            else:
                args.append("%s %s" % (dtype, name))

        for arg in localvars:
            ndim = arg["data"].ndim
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            dname = "d" + name

            dargs.append(dname)

            if ndim > 0:

                shape0 = "".join(["[%d]"%s for s in arg["data"].shape])
                shape1 = ",".join([str(s) for s in arg["data"].shape])

                shapes.append("const int64_t shape_%s[%d] = {%s};" % (name, ndim, shape1))
                args.append("%s %s%s" % (dtype, name, shape0))

            else:
                args.append("%s %s" % (dtype, name))

        return ", ".join(args), ", ".join(dargs), "\n".join(shapes), "\n".join(externs)

    @classmethod
    def _gen_startmain(cls, localvars, modvars):

        dummyargs = []
        main = []
        actualargs = []


        for modname, arg in modvars:
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            if ndim > 0:
                shape = "".join(["[%d]"%s for s in arg["data"].shape])
                main.append(convfmt.format(dtype=dtype, name=name, shape=shape, orgname=modname))
                main.append("%s(*apy_dptr_%s)%s;" % (dtype, name, shape))
                actualargs.append("(*apy_ptr_" + name + ")")

            else:
                actualargs.append("accelpy_var_" + name)


        for arg in localvars:
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            dummyargs.append("void * %s" % name)

            if ndim > 0:
                shape = "".join(["[%d]"%s for s in arg["data"].shape])
                main.append(convfmt.format(dtype=dtype, name=name, shape=shape, orgname=name))
                main.append("%s(*apy_dptr_%s)%s;" % (dtype, name, shape))
                actualargs.append("(*apy_ptr_" + name + ")")

            else:
                actualargs.append("*((%s *) %s)" % (dtype, name))

        dummystr = ", ".join(dummyargs)
        mainstr = "\n".join(main)
        actualstr = ", ".join(actualargs)

        #return "\n".join(argdefs) + "\n\n" + "res = accelpy_kernel(%s);" % ", ".join(startargs)

        return dummystr, mainstr, actualstr


    @classmethod
    def gen_kernelfile(cls, knlhash, dmodname, runid, section, workdir, localvars, modvars):

        kernelpath = os.path.join(workdir, "K%s%s" % (knlhash[2:], cls.srcext))

        kernelargs, launchargs, shapes, externs = cls._gen_kernelargs(localvars, modvars)
        runkernelargs, startmain, actualargs = cls._gen_startmain(localvars, modvars)

        devicekernelargs = ""
        launchconf = ""
        kernelenter = ""
        kernelexit = ""

        kernelparams = {
            "runid": str(runid),
            "externs": externs,
            "macrodefs": cls._gen_macrodefs(localvars, modvars),
            "kernelargs": kernelargs,
            "devicekernelargs": devicekernelargs,
            "runkernelargs": runkernelargs,
            "reshape": shapes,
            "launchconf": launchconf,
            "launchargs": launchargs,
            "kernelenter": kernelenter,
            "kernelexit": kernelexit,
            "spec": "\n".join(section.body),
            "startmain": startmain,
            "actualargs":actualargs 
        }

        with open(kernelpath, "w") as fkernel:
            fkernel.write(kernelsrc.format(**kernelparams))

        import pdb; pdb.set_trace()
        return kernelpath

    @classmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
                        copyin, copyout, alloc, attr):

        datapath = os.path.join(workdir, filename)

        dataparams = {"runid": str(runid), "datamodname": modname}

        modvars = []

        enterargs = []
        enterassign = []
        enterapicall = []
        exitapicall = []

        for cio in copyinout:
            cioname = cio["curname"]
            lcioname = "l" + cioname
            dcioname = "d" + cioname
            dtype = get_c_dtype(cio)

            enterargs.append("void * " + lcioname)

            if cio["data"].ndim > 0:
                #modvars.append("%s * %s;" % (dtype, cioname))
                #modvars.append("%s * %s;" % (dtype, dcioname))
                #enterassign.append("%s = (%s *) %s;" % (cioname, dtype, lcioname))
                modvars.append("void * %s;" % cioname)
                modvars.append("hipDeviceptr_t %s;" % dcioname)
                enterassign.append("%s = %s;" % (cioname, lcioname))
                enterapicall.append(cls._mapto(cioname, dcioname, cio["data"].size, dtype))
                exitapicall.append(cls._mapfrom(cioname, dcioname, cio["data"].size, dtype))

            else:
                modvars.append("%s %s;" % (dtype, cioname))
                enterassign.append("%s = *(%s *) %s;" % (cioname, dtype, lcioname))
                #enterapicall.append(cls._mapto(cionames))
                #exitapicall.append(cls._mapfrom(cionames))

        for ci in copyin:
            ciname = ci["curname"]
            lciname = "l" + ciname
            dciname = "d" + ciname
            dtype = get_c_dtype(ci)

            enterargs.append("void * " + lciname)

            if ci["data"].ndim > 0:
                #modvars.append("%s * %s;" % (dtype, ciname))
                #modvars.append("%s * %s;" % (dtype, dciname))
                #enterassign.append("%s = (%s *) %s;" % (ciname, dtype, lciname))
                modvars.append("void * %s;" % ciname)
                modvars.append("hipDeviceptr_t %s;" % dciname)
                enterassign.append("%s = %s;" % (ciname, lciname))
                enterapicall.append(cls._mapto(ciname, dciname, ci["data"].size, dtype))

            else:
                enterassign.append("%s = *(%s *) %s;" % (ciname, dtype, lciname))
                modvars.append("%s %s;" % (dtype, ciname))
                #enterapicall.append(cls._mapto(cinames))

        for co in copyout:
            coname = co["curname"]
            lconame = "l" + coname
            dconame = "d" + coname
            dtype = get_c_dtype(co)

            enterargs.append("void * " + lconame)

            if co["data"].ndim > 0:
                #modvars.append("%s * %s;" % (dtype, coname))
                #modvars.append("%s * %s;" % (dtype, dconame))
                #enterassign.append("%s = (%s *) %s;" % (coname, dtype, lconame))
                modvars.append("void * %s;" % coname)
                modvars.append("hipDeviceptr_t %s;" % dconame)
                enterassign.append("%s = %s;" % (coname, lconame))
                exitapicall.append(cls._mapfrom(coname, dconame, co["data"].size, dtype))

            else:
                modvars.append("%s %s;" % (dtype, coname))
                enterassign.append("%s = *(%s *) %s;" % (coname, dtype, lconame))
                #exitapicall.append(cls._mapfrom(conames))

        for al in alloc:
            alname = al["curname"]
            lalname = "l" + alname
            dalname = "d" + alname
            dtype = get_c_dtype(al)

            enterargs.append("void * " + lalname)

            if al["data"].ndim > 0:
                #modvars.append("%s * %s;" % (dtype, alname))
                #modvars.append("%s * %s;" % (dtype, dalname))
                #enterassign.append("%s = (%s *) %s;" % (alname, dtype, lalname))
                modvars.append("void * %s;" % alname)
                modvars.append("hipDeviceptr_t %s;" % dalname)
                enterassign.append("%s = %s;" % (alname, lalname))
                enterapicall.append(cls._mapalloc(dciname, ci["data"].size, dtype))

            else:
                modvars.append("%s %s;" % (dtype, alname))
                enterassign.append("%s = *(%s *) %s;" % (alname, dtype, "l"+alname))
                #enterapicall.append(cls._mapalloc(alnames))

        dataparams["modvars"] = "\n".join(modvars)
        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["enterapicall"] =  "\n".join(enterapicall)
        dataparams["enterassign"] = "\n".join(enterassign)
        dataparams["exitargs"] = ""
        dataparams["exitapicall"] = "\n".join(exitapicall)

        with open(datapath, "w") as fdata:
            fdata.write(datasrc.format(**dataparams))

        #import pdb; pdb.set_trace()
        return datapath

#class CppAccel(CppAccelBase):
#    accel = "cpp"
#
#    @classmethod
#    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
#                        copyin, copyout, alloc, attr):
#
#        datapath = os.path.join(workdir, filename)
#
#        dataparams = {"runid": str(runid), "datamodname": modname}
#
#        modvars = []
#
#        enterargs = []
#        enterassign = []
#
#        for item in copyinout+copyin+copyout+alloc:
#            itemname = item["curname"]
#            dtype = get_c_dtype(item)
#
#            modvars.append("%s * %s;" % (dtype, itemname))
#            enterargs.append("void * l"+itemname)
#
#            if item["data"].ndim > 0:
#                enterassign.append("%s = (%s *) %s;" % (itemname, dtype, "l"+itemname))
#
#            else:
#                enterassign.append("%s = *(%s *) %s;" % (itemname, dtype, "l"+itemname))
#
#        dataparams["modvars"] = "\n".join(modvars)
#        dataparams["enterargs"] = ", ".join(enterargs)
#        dataparams["enterdirective"] = ""
#        dataparams["enterassign"] = "\n".join(enterassign)
#        dataparams["exitargs"] = ""
#        dataparams["exitdirective"] = ""
#
#        with open(datapath, "w") as fdata:
#            fdata.write(datasrc.format(**dataparams))
#
#        #import pdb; pdb.set_trace()
#        return datapath



class CudaAccel(CudaHipAccelBase):
    accel = "cuda"

    @classmethod
    def _mapto(cls, names):
        return "#pragma omp target enter data map(to:" + ", ".join(names) + ")"

    @classmethod
    def _mapfrom(cls, names):
        return "#pragma omp target exit data map(from:" + ", ".join(names) + ")"

    @classmethod
    def _mapalloc(cls, names):
        return "#pragma omp target enter data map(alloc:" + ", ".join(names) + ")"


class HipAccel(CudaHipAccelBase):
    accel = "hip"

    @classmethod
    def _mapto(cls, vname, dname, size, tname):

        fmt = ("hipMalloc((void **)&{dname}, {size} * sizeof({type}));\n"
               "hipMemcpyHtoD({dname}, {name}, {size} * sizeof({type}));\n")

        return fmt.format(name=vname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapfrom(cls, vname, dname, size, tname):

        fmt = ("hipMemcpyDtoH({dname}, {name}, {size} * sizeof({type}));\n"
               "hipFree({dname});\n")

        return fmt.format(name=vname, dname=dname, size=str(size), type=tname)

    @classmethod
    def _mapalloc(cls, dname, size, tname):

        fmt = "hipMalloc((void **)&{dname}, {size} * sizeof({type}));\n"

        return fmt.format(dname=dname, size=str(size), type=tname)


_chaccels = OrderedDict()
AccelBase.avails[CudaHipAccelBase.lang] = _chaccels

_chaccels[CudaAccel.accel] = CudaAccel
_chaccels[HipAccel.accel] = HipAccel

