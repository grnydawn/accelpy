"""accelpy Fortran Accelerator module"""

from uuid import uuid4
from accelpy.accel import AccelDataBase
from accelpy.kernel import KernelBase
from accelpy.util import fortline_pack, f_dtypemap
from ctypes import c_int32, c_int64, c_float, c_double


##########################
#  Kernel Code templates
##########################

t_module = """
MODULE {modname}
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

{datavars}

END MODULE
"""

t_main = """
{include}

{varmap}

INTEGER (C_INT64_T) FUNCTION accelpy_start() BIND(C, name="accelpy_start")

    USE, INTRINSIC :: ISO_C_BINDING
    USE {modname}, ONLY : {usevarnames}
    IMPLICIT NONE

    accelpy_start = accelpy_kernel({usevarnames})

CONTAINS

{kernel}

END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_stop() BIND(C, name="accelpy_stop")
    USE, INTRINSIC :: ISO_C_BINDING
    USE {modname}, ONLY : {usevarnames}
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
    USE {modname}, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION({bound}), INTENT(IN), TARGET :: data

    {varname} => data

    {funcname} = 0

END FUNCTION
"""

t_varmap_scalar = """
INTEGER (C_INT64_T) FUNCTION {funcname} (data) BIND(C, name="{funcname}")
    USE, INTRINSIC :: ISO_C_BINDING
    USE {modname}, ONLY : {varname}
    IMPLICIT NONE

    {dtype}, DIMENSION(1), INTENT(IN) :: data

    {varname} = data(1)

    {funcname} = 0

END FUNCTION
"""

class FortranKernel(KernelBase):

    name = "fortran"
    lang = "fortran"

#    dtypemap = {
#        "int32": ["INTEGER (C_INT32_T)", c_int32],
#        "int64": ["INTEGER (C_INT64_T)", c_int64],
#        "float32": ["REAL (C_FLOAT)", c_float],
#        "float64": ["REAL (C_DOUBLE)", c_double]
#    }

    def gen_code(self, compiler):

        macros = {}
        includes = self.add_includes()

        modname  = "mod_" + uuid4().hex[:10]

        module_fmt = {
            "modname": modname,
            "datavars": self._get_datavars(),
        }
        module = t_module.format(**module_fmt)

        main_fmt = {
            "modname":modname,
            "varmap":self._gen_varmap(modname),
            "kernel":self._gen_kernel(),
            "usevarnames":self._gen_usevars(),
            "include":self.get_include(),
        }
        main = t_main.format(**main_fmt)

        #print(module)
        #print(main)
        #import pdb; pdb.set_trace()

        return includes + [module, main], macros

    def get_dtype(self, arg):
        return f_dtypemap[arg["data"].dtype.name][0]

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

    def _gen_varmap(self, modname):

        out = []

        for arg in self.data:
            dtype = self.get_dtype(arg)
            funcname = self.getname_varmap(arg)
            ndim = arg["data"].ndim

            if ndim > 0:
                bound = ",".join([str(s) for s in arg["data"].shape])
                out.append(t_varmap.format(funcname=funcname, varname=arg["curname"],
                            modname=modname, bound=bound, dtype=dtype))
            else:
                out.append(t_varmap_scalar.format(funcname=funcname, modname=modname,
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

class OpenmpFortranKernel(FortranKernel):
    name = "openmp_fortran"

class OpenaccFortranKernel(FortranKernel):
    name = "openacc_fortran"

class OmptargetFortranKernel(OpenmpFortranKernel):
    name = "omptarget_fortran"

OmptargetFortranKernel.avails[OmptargetFortranKernel.name] = OmptargetFortranKernel
OpenaccFortranKernel.avails[OpenaccFortranKernel.name] = OpenaccFortranKernel
OpenmpFortranKernel.avails[OpenmpFortranKernel.name] = OpenmpFortranKernel
FortranKernel.avails[FortranKernel.name] = FortranKernel



##########################
#  AccelData Code templates
##########################

t_ompdata = """
MODULE {modname}
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

{modvardecls}

PUBLIC {modpublics}

CONTAINS

INTEGER (C_INT64_T) FUNCTION dataenter({enterargs}) BIND(C, name="dataenter")
    USE, INTRINSIC :: ISO_C_BINDING

    {entertypedecls}

    {enterassigns}

    !$omp target enter data {entermaps}

    dataenter = 0

END FUNCTION

INTEGER (C_INT64_T) FUNCTION dataexit() BIND(C, name="dataexit")
    USE, INTRINSIC :: ISO_C_BINDING

    !$omp target exit data {exitmaps}

    dataexit = 0

END FUNCTION

END MODULE
"""

class FortranAccelData(AccelDataBase):

    name = "fortran"
    lang = "fortran"

#    dtypemap = {
#        "int32": ["INTEGER (C_INT32_T)", c_int32],
#        "int64": ["INTEGER (C_INT64_T)", c_int64],
#        "float32": ["REAL (C_FLOAT)", c_float],
#        "float64": ["REAL (C_DOUBLE)", c_double]
#    }

    def get_dtype(self, arg):
        return f_dtypemap[arg["data"].dtype.name][0]

    def gen_code(self, compiler):
        pass


class OpenaccFortranAccelData(FortranAccelData):
    name = "openacc_fortran"

class OmptargetFortranAccelData(FortranAccelData):
    name = "omptarget_fortran"

    def gen_code(self, compiler):


        macros = {}

        modname  = "mod_" + uuid4().hex[:10]

        modvardecls = []
        modpublics = []
        enterargs = []
        entertypedecls = []
        enterassigns = []
        mapto = []
        maptofrom = []
        mapalloc = []
        exitfrom = []

        for item in self.mapto:
            ndim = item["data"].ndim
            dtype = self.get_dtype(item)
            lname = "l" + str(item["id"])
            gname = "g" + str(item["id"])

            enterargs.append(lname)
            mapto.append(gname)
            #bound = ",".join([":"]*ndim)
            bound = ",".join([str(s) for s in item["data"].shape])
            entertypedecls.append("%s, DIMENSION(%s), INTENT(INOUT), TARGET :: %s" % (
                            dtype, bound, lname))

            modpublics.append(gname)
            bound = ",".join([":"]*ndim)
            modvardecls.append("%s, DIMENSION(%s), POINTER :: %s" % (
                            dtype, bound, gname))

            enterassigns.append("%s => %s" % (gname, lname))

        strto = "map(to: %s)" % ", ".join(mapto) if mapto else ""

        for item in self.maptofrom:
            ndim = item["data"].ndim
            dtype = self.get_dtype(item)
            lname = "l" + str(item["id"])
            gname = "g" + str(item["id"])

            enterargs.append(lname)
            maptofrom.append(gname)
            bound = ",".join([":"]*ndim)
            entertypedecls.append("%s, DIMENSION(%s), INTENT(INOUT), TARGET :: %s" % (
                            dtype, bound, lname))

            modpublics.append(gname)
            modvardecls.append("%s, DIMENSION(%s), POINTER :: %s" % (
                            dtype, bound, gname))

            enterassigns.append("%s => %s" % (gname, lname))

        strtofrom = "map(tofrom: %s)" % ", ".join(maptofrom) if maptofrom else ""

        for item in self.mapalloc:
            ndim = item["data"].ndim
            dtype = self.get_dtype(item)
            lname = "l" + str(item["id"])
            gname = "g" + str(item["id"])

            enterargs.append(lname)
            mapalloc.append(gname)
            bound = ",".join([":"]*ndim)
            entertypedecls.append("%s, DIMENSION(%s), INTENT(INOUT), TARGET :: %s" % (
                            dtype, bound, lname))

            modpublics.append(gname)
            modvardecls.append("%s, DIMENSION(%s), POINTER :: %s" % (
                            dtype, bound, gname))

            enterassigns.append("%s => %s" % (gname, lname))

        for item in self.mapfrom:
            ndim = item["data"].ndim
            dtype = self.get_dtype(item)
            lname = "l" + str(item["id"])
            gname = "g" + str(item["id"])

            enterargs.append(lname)
            exitfrom.append(gname)
            mapalloc.append(gname)
            #bound = ",".join([":"]*ndim)
            bound = ",".join([str(s) for s in item["data"].shape])
            entertypedecls.append("%s, DIMENSION(%s), INTENT(INOUT), TARGET :: %s" % (
                            dtype, bound, lname))

            modpublics.append(gname)
            bound = ",".join([":"]*ndim)
            modvardecls.append("%s, DIMENSION(%s), POINTER :: %s" % (
                            dtype, bound, gname))

            enterassigns.append("%s => %s" % (gname, lname))

        stralloc = "map(alloc: %s)" % ", ".join(mapalloc) if mapalloc else ""

        entermaps = "%s %s %s" % (strto, strtofrom, stralloc)
        exitmaps = "map(from: %s)" % ", ".join(exitfrom) if exitfrom else ""

        acceldata_fmt = {
            "modname": modname,
            "modvardecls": "\n".join(modvardecls),
            "modpublics": ", ".join(modpublics),
            "enterargs": ", ".join(enterargs),
            "entertypedecls": "\n".join(entertypedecls),
            "enterassigns": "\n".join(enterassigns),
            "entermaps": entermaps,
            "exitmaps": exitmaps,
        }
        ompdata = t_ompdata.format(**acceldata_fmt)

        #print(ompdata)
        #import pdb; pdb.set_trace()

        return [ompdata], macros


    def _get_enterinfo(self):

        enterargs = []
        entertypedecls = []

        for item in self.mapto+self.maptofrom+self.mapalloc:

            ndim = item["data"].ndim
            dtype = self.get_dtype(item)
            varname = "v" + str(item["id"])

            enterargs.append(varname)

            bound = ",".join([":"]*ndim)
            entertypedecls.append("%s, DIMENSION(%s), INTENT(INOUT) :: %s" % (
                            dtype, bound, varname))

        mapto = []
        maptofrom = []
        mapalloc = []

        for item in self.mapto:
            varname = "v" + str(item["id"])
            mapto.append(varname)
        strto = "map(to: %s)" % ", ".join(mapto) if mapto else ""

        for item in self.maptofrom:
            varname = "v" + str(item["id"])
            maptofrom.append(varname)
        strtofrom = "map(tofrom: %s)" % ", ".join(maptofrom) if maptofrom else ""

        for item in self.mapalloc:
            varname = "v" + str(item["id"])
            mapalloc.append(varname)
        stralloc = "map(alloc: %s)" % ", ".join(mapalloc) if mapalloc else ""

        entermaps = "%s %s %s" % (strto, strtofrom, stralloc)

        return ", ".join(enterargs), "\n".join(entertypedecls), entermaps


    def _get_exitinfo(self):

        exitargs = []
        exittypedecls = []

        for item in self.mapfrom+self.maprelease+self.mapdelete:

            ndim = item["data"].ndim
            dtype = self.get_dtype(item)
            varname = "v" + str(item["id"])

            exitargs.append(varname)

            bound = ",".join([":"]*ndim)
            exittypedecls.append("%s, DIMENSION(%s), INTENT(INOUT) :: %s" % (
                            dtype, bound, varname))

        mapfrom = []
        maprelease = []
        mapdelete = []

        for item in self.mapfrom:
            varname = "v" + str(item["id"])
            mapfrom.append(varname)
        strfrom = "map(from: %s)" % ", ".join(mapfrom) if mapfrom else ""

        for item in self.maprelease:
            varname = "v" + str(item["id"])
            maprelease.append(varname)
        strrelease = "map(release: %s)" % ", ".join(maprelease) if maprelease else ""

        for item in self.mapdelete:
            varname = "v" + str(item["id"])
            mapdelete.append(varname)
        strdelete = "map(delete: %s)" % ", ".join(mapdelete) if mapdelete else ""

        exitmaps = "%s %s %s" % (strfrom, strrelease, strdelete)

        return ", ".join(exitargs), "\n".join(exittypedecls), exitmaps

AccelDataBase.avails[OmptargetFortranAccelData.name] = OmptargetFortranAccelData
AccelDataBase.avails[OpenaccFortranAccelData.name] = OpenaccFortranAccelData
