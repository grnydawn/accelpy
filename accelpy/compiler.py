"""accelpy Compiler module"""

import os, sys, abc, time, threading, inspect, hashlib
from numpy import ndarray
from collections import OrderedDict

from accelpy.util import Object, shellcmd, which, get_config

#########################
# Generic Compilers
#########################

class Compiler(Object):
    """Compiler Base Class"""

    avails = OrderedDict()
    libext = "so"
    objext = "o"
    opt_compile_only = "-c"
    opt_debug = "-O0 -g"

    def __init__(self, path, option=None):

        self.version = []

        if isinstance(path, str):
            self.set_version(path)
            self.path = path

        elif isinstance(path, (list, tuple)):
            for p in path:
                try:
                    self.set_version(path)
                    self.path = path
                    break
                except:
                    pass

            assert self.path
        else:
            raise Exception("Unsupported compiler path type" % str(path))

        opts = self.get_option()
        self.option = option.format(default=opts) if option else opts

        if sys.platform == "darwin":
            self.libext = "dylib"

    @abc.abstractmethod
    def parse_version(self, stdout):
        pass

    @abc.abstractmethod
    def get_option(self):
        return ""

    def from_path(self, path):
        import pdb; pdb.set_trace()

    def set_version(self, path=None):
        out = shellcmd((path if path else self.path) + " " + self.opt_version)

        if out.returncode != 0:
            raise Exception("Compiler version check fails: %s" % out.stderr)

        self.version = self.parse_version(out.stdout)

    def _compile(self, code, ext, macros, debug):

        _macros = []
        for k, v in macros.items():
            _macros.append("-D %s=\"%s\"" % (k, str(v)) if v else ("-D "+str(k)))

        opt_macro = " ".join(_macros)
        text = (code + opt_macro + self.vendor + "".join(self.version) + ext)
        name =  hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

        blddir = get_config("blddir")
        filename = name + "." + self.codeext
        codepath = os.path.join(blddir, filename)
        outfile = os.path.join(blddir, name + "." + ext)

        fortmod = os.path.join(blddir, "accelpy_global.mod")
        if code.lstrip().startswith("MODULE accelpy_global") and os.path.isfile(fortmod):
            os.remove(fortmod)
            if os.path.isfile(outfile):
                os.remove(outfile)

        if not os.path.isfile(codepath):
            try:
                with open(codepath, "w") as f:
                    f.write(code)

            except FileExistsError:
                pass

        if debug:
            debugdir = "_accelpy_debug_"

            if not os.path.isdir(debugdir):
                try:
                    os.makedirs(debugdir)

                except FileExistsError:
                    pass

            debugfile = os.path.join(debugdir, filename) 

            if not os.path.isdir(debugfile):
                try:
                    with open(debugfile, "w") as f:
                        f.write(code)

                except FileExistsError:
                    pass

        if not os.path.isfile(outfile):

            option = self.opt_debug + " " if debug else ""
            option += self.opt_compile_only + " " + self.get_option()

            # PGI infoopt :  -Minfo=acc
            build_cmd = "{compiler} {option} {macro} -o {outfile} {infile}".format(
                            compiler=self.path, option=option, outfile=outfile,
                            infile=codepath, macro=opt_macro)

            #print(build_cmd)
            #import pdb; pdb.set_trace()

            out = shellcmd(build_cmd)
            #print(str(out.stdout).replace("\\n", "\n"))

            if out.returncode != 0:
                errmsg = str(out.stderr).replace("\\n", "\n")
                raise Exception("Compilation fails: %s" % errmsg)

            if not os.path.isfile(outfile):
                raise Exception("Output is not generated.")

        return outfile

    def _link(self, ext, objfiles):

        objhashes = []
        for objfile in objfiles:
            objhashes.append(os.path.basename(objfile))

        text = (self.vendor + "".join(self.version) + ext + "".join(objhashes))
        name =  hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

        outfile = os.path.join(get_config("blddir"), name + "." + ext)
        option = self.get_option()

        build_cmd = "{compiler} {option} -o {outfile} {objfiles}".format(
                        compiler=self.path, option=option, outfile=outfile,
                        objfiles=" ".join(objfiles))

        #print(build_cmd)
        #import pdb; pdb.set_trace()
        out = shellcmd(build_cmd)

        if out.returncode != 0:
            errmsg = str(out.stderr).replace("\\n", "\n")
            raise Exception("Compilation fails: %s" % errmsg)

        if not os.path.isfile(outfile):
            raise Exception("Output is not generated.")

        return outfile

    def compile(self, codes, macros, debug):

        lib = None

        objfiles = []

        # build object files
        if isinstance(codes, str):
            objfiles.append(self._compile(codes, self.objext, macros, debug))

        elif isinstance(codes, (list, tuple)):
            for code in codes:
                objfiles.append(self._compile(code, self.objext, macros, debug))

        return self._link(self.libext, objfiles)


class CppCompiler(Compiler):

    lang = "cpp"
    codeext = "cpp"


class FortranCompiler(Compiler):

    lang = "fortran"
    codeext = "F90"


class CppCppCompiler(CppCompiler):

    accel = "cpp"


class HipCppCompiler(CppCompiler):

    accel = "hip"


class CudaCppCompiler(CppCompiler):

    accel = "cuda"
    codeext = "cu"


class OpenaccCppCompiler(CppCompiler):

    accel = "openacc_cpp"


class OpenmpCppCompiler(CppCompiler):

    accel = "openmp_cpp"


class FortranFortranCompiler(FortranCompiler):

    accel = "fortran"


class OpenaccFortranCompiler(FortranCompiler):

    accel = "openacc_fortran"


class OpenmpFortranCompiler(FortranCompiler):

    accel = "openmp_fortran"


################
# GNU Compilers
################

class GnuCppCompiler(CppCompiler):

    vendor = "gnu"
    opt_version = "--version"

    def __init__(self, path=None, option=None):

        if not path:
            path = "g++"

        super(GnuCppCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()
        
        if sys.platform == "darwin":
            if items[:3] == [b'Apple', b'clang', b'version']:
                return items[3].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:3]))

        elif sys.platform == "linux":
            if items[:2] == [b'g++', b'(GCC)']:
                return items[2].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:2]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "darwin":
            opts = "-dynamiclib -fPIC " + super(GnuCppCompiler, self).get_option()

        elif sys.platform == "linux":
            opts = "-shared -fPIC " + super(GnuCppCompiler, self).get_option()

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts

class GnuCppCppCompiler(CppCppCompiler, GnuCppCompiler):
    pass


class GnuFortranCompiler(FortranCompiler):

    vendor = "gnu"
    opt_version = "--version"
    opt_moddir = "-J %s"

    def __init__(self, path=None, option=None):

        if not path:
            path = "gfortran"

        super(GnuFortranCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()
        
        if sys.platform in ("darwin", "linux"):
            if items[:2] == [b'GNU', b'Fortran']:
                return items[3].decode().split(".")
            raise Exception("Unknown compiler version syntax: %s" % str(items[:3]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        moddir = self.opt_moddir % get_config("blddir")

        if sys.platform == "darwin":
            opts = ("-dynamiclib -fPIC %s " % moddir  +
                    super(GnuFortranCompiler, self).get_option())

        elif sys.platform == "linux":
            opts = ("-shared -fPIC %s " % moddir +
                    super(GnuFortranCompiler, self).get_option())

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts


class GnuFortranFortranCompiler(FortranFortranCompiler, GnuFortranCompiler):
        pass


class GnuOpenaccCppCompiler(OpenaccCppCompiler, GnuCppCompiler):

    def get_option(self):
        return "-fopenacc " + super(GnuOpenaccCppCompiler, self).get_option()


class GnuOpenmpCppCompiler(OpenmpCppCompiler, GnuCppCompiler):

    def get_option(self):
        return "-fopenmp " + super(GnuOpenmpCppCompiler, self).get_option()


class GnuOpenaccFortranCompiler(OpenaccFortranCompiler, GnuFortranCompiler):

    def get_option(self):
        return "-fopenacc " + super(GnuOpenaccFortranCompiler, self).get_option()


class GnuOpenmpFortranCompiler(OpenmpFortranCompiler, GnuFortranCompiler):

    def get_option(self):
        return "-fopenmp " + super(GnuOpenmpFortranCompiler, self).get_option()


################
# Cray Compilers
################

class CrayClangCppCompiler(CppCompiler):

    vendor = "cray"
    opt_version = "--version"
    #opt_openmp = "-h omp"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("CC"):
            path = "CC"

        elif which("crayCC"):
            path = "crayCC"

        elif which("clang++"):
            path = "clang++"

        super(CrayClangCppCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()

        if sys.platform == "linux":
            if items[:3] == [b'Cray', b'clang', b'version']:
                return items[3].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:3]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "linux":
            opts = "-shared -fPIC " + super(CrayClangCppCompiler, self).get_option()

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts


class CrayClangCppCppCompiler(CppCppCompiler, CrayClangCppCompiler):
    pass


class CrayClangOpenaccCppCompiler(OpenaccCppCompiler, CrayClangCppCompiler):

    def get_option(self):
        return "-h pragma=acc " + super(CrayClangOpenaccCppCompiler, self).get_option()


class CrayClangOpenmpCppCompiler(OpenmpCppCompiler, CrayClangCppCompiler):

    def get_option(self):
        return "-h pragma=omp " + super(CrayClangOpenmpCppCompiler, self).get_option()


class CrayFortranCompiler(FortranCompiler):

    vendor = "cray"
    opt_version = "--version"
    #opt_openmp = "-h omp"
    opt_moddir = "-J %s"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("ftn"):
            path = "ftn"

        elif which("crayftn"):
            path = "crayftn"

        super(CrayFortranCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()
        
        if sys.platform == "linux":
            if items[:4] == [b'Cray', b'Fortran', b':', b'Version']:
                return items[4].decode().split(".")
            raise Exception("Unknown compiler version syntax: %s" % str(items[:4]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "linux":
            moddir = self.opt_moddir % get_config("blddir")
            opts = ("-shared -fPIC %s " % moddir +
                    super(CrayFortranCompiler, self).get_option())
        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts


class CrayFortranFortranCompiler(FortranFortranCompiler, CrayFortranCompiler):
    pass


class CrayOpenaccFortranCompiler(OpenaccFortranCompiler, CrayFortranCompiler):

    def get_option(self):
        return "-h acc,noomp " + super(CrayOpenaccFortranCompiler, self).get_option()

class CrayOpenmpFortranCompiler(OpenmpFortranCompiler, CrayFortranCompiler):

    def get_option(self):
        return "-h omp,noacc " + super(CrayOpenmpFortranCompiler, self).get_option()


################
# AMD Compilers
################

class AmdClangCppCompiler(CppCompiler):

    vendor = "amd"
    opt_version = "--version"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("amdclang++"):
            path = "amdclang++"

        super(AmdClangCppCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()

        if sys.platform == "linux":
            if items[:3] == [b'AMD', b'clang', b'version']:
                return items[2].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:2]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "linux":
            opts = "-shared -fPIC " + super(AmdClangCppCompiler, self).get_option()

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts


class AmdClangCppCppCompiler(CppCppCompiler, AmdClangCppCompiler):
    pass


class AmdClangOpenmpCppCompiler(OpenmpCppCompiler, AmdClangCppCompiler):

    def get_option(self):
        return "-fopenmp " + super(AmdClangOpenmpCppCompiler, self).get_option()


class AmdFlangFortranCompiler(FortranCompiler):

    vendor = "amd"
    opt_version = "--version"
    opt_moddir = "-J %s"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("amdflang"):
            path = "amdflang"

        super(AmdFlangFortranCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()
        
        if sys.platform == "linux":
            if items[:3] == [b'AMD', b'flang-new', b'version']:
                return items[2].decode().split(".")
            raise Exception("Unknown compiler version syntax: %s" % str(items[:2]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "linux":
            moddir = self.opt_moddir % get_config("blddir")
            opts = ("-shared -fPIC %s " % moddir +
                    super(AmdFlangFortranCompiler, self).get_option())

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts


class AmdFlangFortranFortranCompiler(FortranFortranCompiler, AmdFlangFortranCompiler):
    pass


class AmdFlangOpenmpFortranCompiler(OpenmpFortranCompiler, AmdFlangFortranCompiler):

    def get_option(self):
        return "-fopenmp " + super(AmdFlangOpenmpFortranCompiler, self).get_option()


class AmdHipCppCompiler(HipCppCompiler):

    vendor = "amd"
    opt_version = "--version"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("hipcc"):
            path = "hipcc"

        super(AmdHipCppCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()

        if sys.platform == "linux":
            if items[:2] == [b'HIP', b'version:']:
                return items[2].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:2]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "linux":
            opts = "-shared -fPIC " + super(AmdHipCppCompiler, self).get_option()

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts

###################
# IBM XL Compilers
###################

class IbmXlCppCompiler(CppCompiler):

    vendor = "ibm"
    opt_version = "-qversion"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("xlc++_r"):
            path = "xlc++_r"

        elif which("xlc++"):
            path = "xlc++"

        super(IbmXlCppCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()

        if sys.platform == "linux":
            if items[:3] == [b'IBM', b'XL', b'C/C++']:
                return items[5].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:3]))

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

    def get_option(self):

        if sys.platform == "linux":
            opts = "-shared -fPIC " + super(IbmXlCppCompiler, self).get_option()

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts


class IbmXlCppCppCompiler(CppCppCompiler, IbmXlCppCompiler):
    pass


class IbmXlOpenmpCppCompiler(OpenmpCppCompiler, IbmXlCppCompiler):

    def get_option(self):
        return "-qsmp=omp " + super(IbmXlOpenmpCppCompiler, self).get_option()


class IbmXlFortranCompiler(FortranCompiler):

    vendor = "ibm"
    opt_moddir = "-qmoddir=%s"
    opt_version = "-qversion"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("xlf2008_r"):
            path = "xlf2008_r"

        elif which("xlf2008"):
            path = "xlf2008"

        elif which("xlf2003_r"):
            path = "xlf2003_r"

        elif which("xlf2003"):
            path = "xlf2003"

        elif which("xlf95_r"):
            path = "xlf95_r"

        elif which("xlf95"):
            path = "xlf95"

        elif which("xlf90_r"):
            path = "xlf90_r"

        elif which("xlf90"):
            path = "xlf90"

        super(IbmXlFortranCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()
        
        if sys.platform == "linux":
            if items[:3] == [b'IBM', b'XL', b'Fortran']:
                return items[5].decode().split(".")
            raise Exception("Unknown compiler version syntax: %s" % str(items[:3]))

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

    def get_option(self):

        if sys.platform == "linux":
            moddir = self.opt_moddir % get_config("blddir")
            opts = ("-qmkshrobj -qpic %s " % moddir +
                    super(IbmXlFortranCompiler, self).get_option())

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts


class IbmXlFortranFortranCompiler(FortranFortranCompiler, IbmXlFortranCompiler):
    pass


class IbmXlOpenmpFortranCompiler(OpenmpFortranCompiler, IbmXlFortranCompiler):

    def get_option(self):
        return "-qsmp=omp " + super(IbmXlOpenmpFortranCompiler, self).get_option()


###################
# Nvidia Compilers
###################

class NvidiaCudaCppCompiler(CudaCppCompiler):

    vendor = "nvidia"
    opt_version = "--version"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("nvcc"):
            path = "nvcc"

        super(NvidiaCudaCppCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()

        if sys.platform == "linux":
            if items[:2] == [b'nvcc:', b'NVIDIA']:
                idx = items.index(b'Build')
                return items[idx+1].decode().split("_")
            raise Exception("Unknown version syntaxt: %s" % str(items[:2]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "linux":
            opts = "-shared --compiler-options '-fPIC' " + super(NvidiaCudaCppCompiler, self).get_option()

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts


###################
# PGI Compilers
###################

class PgiCppCompiler(CppCompiler):

    vendor = "pgi"
    opt_version = "--version"
    #opt_openmp = "-mp"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("pgc++"):
            path = "pgc++"

        super(PgiCppCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.strip().split()

        if sys.platform == "linux":
            if items[0] == b'pgc++':
                return items[1].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:1]))

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)


    def get_option(self):

        if sys.platform == "linux":
            opts = "-shared -fpic " + super(PgiCppCompiler, self).get_option()

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts


class PgiFortranCompiler(FortranCompiler):

    vendor = "pgi"
    opt_version = "--version"
    opt_moddir = "-module %s"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("pgfortran"):
            path = "pgfortran"

        super(PgiFortranCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.strip().split()

        if sys.platform == "linux":
            if items[0] == b'pgfortran':
                return items[1].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:1]))

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

    def get_option(self):

        if sys.platform == "linux":
            moddir = self.opt_moddir % get_config("blddir")
            opts = ("-shared -fpic %s " % moddir +
                    super(PgiFortranCompiler, self).get_option())

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts


class PgiCppCppCompiler(CppCppCompiler, PgiCppCompiler):
    pass


class PgiFortranFortranCompiler(FortranFortranCompiler, PgiFortranCompiler):
    pass


class PgiOpenaccCppCompiler(OpenaccCppCompiler, PgiCppCompiler):

    def get_option(self):
        return "-acc " + super(PgiOpenaccCppCompiler, self).get_option()


class PgiOpenaccFortranCompiler(OpenaccFortranCompiler, PgiFortranCompiler):

    def get_option(self):
        return "-acc " + super(PgiOpenaccFortranCompiler, self).get_option()


class PgiOpenmpCppCompiler(OpenmpCppCompiler, PgiCppCompiler):

    def get_option(self):
        return "-mp " + super(PgiOpenmpCppCompiler, self).get_option()


class PgiOpenmpFortranCompiler(OpenmpFortranCompiler, PgiFortranCompiler):

    def get_option(self):
        return "-mp " + super(PgiOpenmpFortranCompiler, self).get_option()


###################
# Intel Compilers
###################

class IntelCppCompiler(CppCompiler):

    vendor = "intel"
    opt_version = "--version"
    #opt_openmp = "-qopenmp"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("icpc"):
            path = "icpc"

        super(IntelCppCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.strip().split()

        if sys.platform == "linux":
            if items[:2] == [b'icpc', b'(ICC)']:
                return items[2].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:1]))

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

    def get_option(self):

        if sys.platform == "linux":
            opts = "-shared -fpic " + super(IntelCppCompiler, self).get_option()

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts


class IntelCppCppCompiler(CppCppCompiler, IntelCppCompiler):
    pass



class IntelFortranCompiler(FortranCompiler):

    vendor = "intel"
    opt_version = "--version"
    #opt_openmp = "-qopenmp"
    opt_moddir = "-module %s"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("ifort"):
            path = "ifort"

        super(IntelFortranCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.strip().split()

        if sys.platform == "linux":
            if items[:2] == [b'ifort', b'(IFORT)']:
                return items[2].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:1]))

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

    def get_option(self):

        if sys.platform == "linux":
            moddir = self.opt_moddir % get_config("blddir")
            return ("-shared -fpic %s " % moddir +
                    super(IntelFortranCompiler, self).get_option())
        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts


class IntelFortranFortranCompiler(FortranFortranCompiler, IntelFortranCompiler):
    pass


# priorities
_lang_priority = ["fortran", "cpp"]
_accel_priority = ["openmp_fortran", "openmp_cpp", "openacc_fortran",
                    "openacc_cpp", "hip", "cuda", "fortran", "cpp"]
_vendor_priority = ["cray", "amd", "nvidia", "intel", "pgi", "ibm", "gnu"]

def sort_compilers():

    def _langsort(l):
        return _lang_priority.index(l.lang)

    def _accelsort(a):
        if hasattr(a, "accel"):
            return _accel_priority.index(a.accel)

        else:
            return -1

    def _vendorsort(v):
        return _vendor_priority.index(v.vendor)

    Compiler.avails = OrderedDict()

    for langsubc in sorted(Compiler.__subclasses__(), key=_langsort):
        lang = langsubc.lang
        Compiler.avails[lang] = OrderedDict()

        for accelsubc in sorted(langsubc.__subclasses__(), key=_accelsort):
            if not hasattr(accelsubc, "accel"):
                continue

            accel = accelsubc.accel
            Compiler.avails[lang][accel] = OrderedDict()

            for vendorsubc in sorted(accelsubc.__subclasses__(), key=_vendorsort):
                vendor = vendorsubc.vendor
                Compiler.avails[lang][accel][vendor] = vendorsubc

sort_compilers()


#CRAY	
#
#MODULES: PrgEnv-cray craype-accel-nvidia35 cudatoolkit (libsci_acc is automatically included) craype-accel-host for host cpu code 
#FORTRAN FLAGS: -h acc,noomp #OpenACC -h noacc,omp #OpenMP4.x -fpic -dynamic -lcudart -G2
#C FLAGLS: -h pragma=acc -h nopragma=omp -fpic -dynamic -lcudart -Gp 
#HELP FLAGS: -rm -h msgs
#
# 
#PGI	
#
#MODULES: PrgEnv-pgi cudatoolkit
#FORTRAN FLAGSS: -acc -ta=nvidia or -ta=multicore for host cpu code -lcudart -mcmodel=medium
#C FLAGS: -acc -ta=nvidia -lcudart -mcmodel=medium
#HELP FLAGS: -Minfo=accel
