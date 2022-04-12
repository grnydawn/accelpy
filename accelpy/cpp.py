"""accelpy Cpp-based Accelerator module"""

import os

from collections import OrderedDict

from accelpy.util import get_c_dtype
from accelpy.accel import AccelBase


convfmt = "{dtype}(*apy_ptr_{name}){shape} = reinterpret_cast<{dtype}(*){shape}>({orgname});"

datasrc = """
#include <stdint.h>

{modvars}

extern "C" int64_t dataenter_{runid}({enterargs}) {{

    int64_t res;

    {enterassign}

    {enterdirective}

    res = 0;

    return res;

}}

extern "C" int64_t dataexit_{runid}({exitargs}) {{

    int64_t res;

    {exitdirective}

    res = 0;

    return res;

}}
"""

kernelsrc = """
#include <stdio.h>
#include <stdint.h>

{externs}

{macrodefs}

int64_t kernel_{runid}({kernelargs}) {{
    int64_t res;

    {reshape}

    {spec}

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


class CppAccelBase(AccelBase):

    lang = "cpp"
    srcext = ".cpp"

    def _mapto(cls, names):
        raise NotImplementedError("_mapto")

    def _mapfrom(cls, names):
        raise NotImplementedError("_mapfrom")

    def _mapalloc(cls, names):
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
#        macros.append("#define ARG(varname) apy_type_##varname varname apy_shapestr_##varname")
#        macros.append("#define VAR(varname) (*apy_ptr_##varname)")
#        macros.append("#define DVAR(varname) (*apy_dptr_##varname)")
#        macros.append("#define PTR(varname) apy_ptr_##varname")
#        macros.append("#define DPTR(varname) apy_dptr_##varname")
#        macros.append("#define FLATTEN(varname) accelpy_var_##varname")

        for oldname, arg in modvars:
            dtype = get_c_dtype(arg)
            name = arg["curname"]
            ndim = arg["data"].ndim

            consts.append("const int64_t apy_size_%s = %d;" % (name, arg["data"].size))
            typedefs.append("typedef %s apy_type_%s;" % (dtype, name))

            if ndim > 0:

                shapestr = "".join(["[%d]"%s for s in arg["data"].shape])
                #macros.append("#define apy_shapestr_%s %s" % (name, shapestr))
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
                #macros.append("#define apy_shapestr_%s %s" % (name, shapestr))
                for d, s in enumerate(arg["data"].shape):
                    consts.append("const int64_t apy_shape_%s%d = %d;" % (name, d, s))
            else:
                pass

        return "\n".join(macros) + "\n\n" + "\n".join(typedefs) + "\n\n" + "\n".join(consts)

    @classmethod
    def _gen_kernelargs(cls, localvars, modvars):

        args = []
        shapes = []
        externs = []

        for modname, arg in modvars:
            ndim = arg["data"].ndim
            dtype = get_c_dtype(arg)
            name = arg["curname"]

            externs.append("extern void * %s;" % modname)

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

            if ndim > 0:
                shape0 = "".join(["[%d]"%s for s in arg["data"].shape])
                shape1 = ",".join([str(s) for s in arg["data"].shape])

                shapes.append("const int64_t shape_%s[%d] = {%s};" % (name, ndim, shape1))
                args.append("%s %s%s" % (dtype, name, shape0))

            else:
                args.append("%s %s" % (dtype, name))

        return ", ".join(args), "\n".join(shapes), "\n".join(externs)

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

        kernelargs, shapes, externs = cls._gen_kernelargs(localvars, modvars)
        runkernelargs, startmain, actualargs = cls._gen_startmain(localvars, modvars)

        kernelparams = {
            "runid": str(runid),
            "externs": externs,
            "macrodefs": cls._gen_macrodefs(localvars, modvars),
            "kernelargs": kernelargs,
            "runkernelargs": runkernelargs,
            "reshape": shapes,
            "spec": "\n".join(section.body),
            "startmain": startmain,
            "actualargs":actualargs 
        }

        with open(kernelpath, "w") as fkernel:
            fkernel.write(kernelsrc.format(**kernelparams))

        #import pdb; pdb.set_trace()
        return kernelpath

class CppAccel(CppAccelBase):
    accel = "cpp"

    @classmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
                        copyin, copyout, alloc, attr):

        datapath = os.path.join(workdir, filename)

        dataparams = {"runid": str(runid), "datamodname": modname}

        modvars = []

        enterargs = []
        enterassign = []

        for item in copyinout+copyin+copyout+alloc:
            itemname = item["curname"]
            dtype = get_c_dtype(item)

            modvars.append("%s * %s;" % (dtype, itemname))
            enterargs.append("void * l"+itemname)

            if item["data"].ndim > 0:
                enterassign.append("%s = (%s *) %s;" % (itemname, dtype, "l"+itemname))

            else:
                enterassign.append("%s = *(%s *) %s;" % (itemname, dtype, "l"+itemname))

        dataparams["modvars"] = "\n".join(modvars)
        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["enterdirective"] = ""
        dataparams["enterassign"] = "\n".join(enterassign)
        dataparams["exitargs"] = ""
        dataparams["exitdirective"] = ""

        with open(datapath, "w") as fdata:
            fdata.write(datasrc.format(**dataparams))

        #import pdb; pdb.set_trace()
        return datapath



class OpenmpCppAccel(CppAccel):
    accel = "openmp"


class AcctargetCppAccel(CppAccelBase):

    @classmethod
    def gen_datafile(cls, modname, filename, runid, workdir, copyinout,
                        copyin, copyout, alloc, attr):

        datapath = os.path.join(workdir, filename)

        dataparams = {"runid": str(runid), "datamodname": modname}

        modvars = []

        enterargs = []
        enterassign = []
        enterdirective = []
        exitdirective = []

        cionames = []
        cinames = []
        conames = []
        alnames = []

        for cio in copyinout:
            cioname = cio["curname"]
            lcioname = "l" + cioname
            dtype = get_c_dtype(cio)

            enterargs.append("void * " + lcioname)

            if cio["data"].ndim > 0:
                modvars.append("%s * %s;" % (dtype, cioname))
                enterassign.append("%s = (%s *) %s;" % (cioname, dtype, "l"+cioname))
                cionames.append("%s[0:%d]" % (cioname, cio["data"].size))

            else:
                modvars.append("%s %s;" % (dtype, cioname))
                enterassign.append("%s = *(%s *) %s;" % (cioname, dtype, "l"+cioname))
                cionames.append(cioname)

        if cionames:
            enterdirective.append(cls._mapto(cionames))
            exitdirective.append(cls._mapfrom(cionames))


        for ci in copyin:
            ciname = ci["curname"]
            lciname = "l" + ciname
            dtype = get_c_dtype(ci)

            enterargs.append("void * " + lciname)

            if ci["data"].ndim > 0:
                modvars.append("%s * %s;" % (dtype, ciname))
                enterassign.append("%s = (%s *) %s;" % (ciname, dtype, "l"+ciname))
                cinames.append("%s[0:%d]" % (ciname, ci["data"].size))

            else:
                enterassign.append("%s = *(%s *) %s;" % (ciname, dtype, "l"+ciname))
                modvars.append("%s %s;" % (dtype, ciname))
                cinames.append(ciname)

        if cinames:
            enterdirective.append(cls._mapto(cinames))

        for co in copyout:
            coname = co["curname"]
            dtype = get_c_dtype(co)

            enterargs.append("void * l"+coname)

            if co["data"].ndim > 0:
                modvars.append("%s * %s;" % (dtype, coname))
                enterassign.append("%s = (%s *) %s;" % (coname, dtype, "l"+coname))
                conames.append("%s[0:%d]" % (coname, co["data"].size))

            else:
                modvars.append("%s %s;" % (dtype, coname))
                enterassign.append("%s = *(%s *) %s;" % (coname, dtype, "l"+coname))
                conames.append(coname)

        if conames:
            alnames.extend(cionames)
            alnames.extend(conames)
            exitdirective.append(cls._mapfrom(conames))

        for al in alloc:
            alname = al["curname"]
            dtype = get_c_dtype(al)

            enterargs.append("void * l"+alname)

            if al["data"].ndim > 0:
                modvars.append("%s * %s;" % (dtype, alname))
                enterassign.append("%s = (%s *) %s;" % (alname, dtype, "l"+alname))
                alnames.append("%s[0:%d]" % (alname, al["data"].size))

            else:
                modvars.append("%s %s;" % (dtype, alname))
                enterassign.append("%s = *(%s *) %s;" % (alname, dtype, "l"+alname))
                alnames.append("%s" % alname)

        if alnames:
            enterdirective.append(cls._mapalloc(alnames))

        dataparams["modvars"] = "\n".join(modvars)
        dataparams["enterargs"] = ", ".join(enterargs)
        dataparams["enterdirective"] =  "\n".join(enterdirective)
        dataparams["enterassign"] = "\n".join(enterassign)
        dataparams["exitargs"] = ""
        dataparams["exitdirective"] = "\n".join(exitdirective)

        with open(datapath, "w") as fdata:
            fdata.write(datasrc.format(**dataparams))

        #import pdb; pdb.set_trace()
        return datapath


class OmptargetCppAccel(AcctargetCppAccel):
    accel = "omptarget"

    @classmethod
    def _mapto(cls, names):
        return "#pragma omp target enter data map(to:" + ", ".join(names) + ")"

    @classmethod
    def _mapfrom(cls, names):
        return "#pragma omp target exit data map(from:" + ", ".join(names) + ")"

    @classmethod
    def _mapalloc(cls, names):
        return "#pragma omp target enter data map(alloc:" + ", ".join(names) + ")"


class OpenaccCppAccel(AcctargetCppAccel):
    accel = "openacc"

    @classmethod
    def _mapto(cls, names):
        return "#pragma acc enter data copyin(" + ", ".join(names) + ")"

    @classmethod
    def _mapfrom(cls, names):
        return "#pragma acc exit data copyout(" + ", ".join(names) + ")"

    @classmethod
    def _mapalloc(cls, names):
        return "#pragma acc enter data create(" + ", ".join(names) + ")"

_cppaccels = OrderedDict()
AccelBase.avails[CppAccelBase.lang] = _cppaccels

_cppaccels[CppAccel.accel] = CppAccel
_cppaccels[OpenmpCppAccel.accel] = OpenmpCppAccel
_cppaccels[OmptargetCppAccel.accel] = OmptargetCppAccel
_cppaccels[OpenaccCppAccel.accel] = OpenaccCppAccel

