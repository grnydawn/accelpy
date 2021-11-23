"""accelpy Compiler module"""

import os, sys, abc, time, threading, inspect, hashlib
from numpy.ctypeslib import load_library
from numpy import ndarray
from collections import OrderedDict

from accelpy.core import Object
from accelpy.util import shellcmd, which
from accelpy import _config

#########################
# Generic Compilers
#########################

class Compiler(Object):
    """Compiler Base Class"""

    avails = OrderedDict()
    libext = "so"
    objext = "o"
    opt_compile_only = "-c"

    def __init__(self, path, option=None):

        self._blddir = _config["session"]["workdir"]

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

    def _compile(self, code, ext, compile_only=False, objfiles=[], moddir=None):


        objhashes = []
        for objfile in objfiles:
            objhashes.append(os.path.basename(objfile))

        text = (code + self.vendor + "".join(self.version) + ext + "".join(objhashes))
        name =  hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

        codepath = os.path.join(self._blddir, name + "." + self.codeext)
        with open(codepath, "w") as f:
            f.write(code)

        outfile = os.path.join(self._blddir, name + "." + ext)

        if compile_only:
            option = self.opt_compile_only + " " + self.get_option()

        else:
            option = self.get_option()

        if moddir:
            option += " " + self.opt_moddir % moddir

        build_cmd = "{compiler} {option} -o {outfile} {infile} {objfiles}".format(
                        compiler=self.path, option=option, outfile=outfile,
                        infile=codepath, objfiles=" ".join(objfiles))

        #print(build_cmd)
        #import pdb; pdb.set_trace()
        out = shellcmd(build_cmd)

        if out.returncode != 0:
            raise Exception("Compilation fails: %s" % out.stderr)

        if not os.path.isfile(outfile):
            raise Exception("Output is not generated.")

        return outfile

    def compile(self, code):

        lib = None


        for f in os.listdir(self._blddir):
            os.remove(os.path.join(self._blddir, f))

        objfiles = []

        # build object files
        if isinstance(code, str):
            main_code = code

        elif isinstance(code, (list, tuple)):
            main_code = code[-1]

            for extracode in code[:-1]:
                objfiles.append(self._compile(extracode, self.objext,
                                    compile_only=True, moddir=self._blddir))

        libfile = self._compile(main_code, self.libext, objfiles=objfiles)

        libdir, libname = os.path.split(libfile)
        name, _ = os.path.splitext(libname)

        lib = load_library(name, libdir)

        return lib


# TODO: should be abstract
class CppCompiler(Compiler):

    lang = "cpp"
    codeext = "cpp"

# TODO: should be abstract
class FortranCompiler(Compiler):

    lang = "fortran"
    codeext = "F90"


# TODO: should be abstract
class CppCppCompiler(CppCompiler):

    accel = "cpp"
    opt_version = "--version"


# TODO: should be abstract
class FortranFortranCompiler(FortranCompiler):

    accel = "fortran"
    opt_version = "--version"


################
# GNU Compilers
################

class GnuCppCppCompiler(CppCppCompiler):

    vendor = "gnu"
    opt_openmp = "--fopenmp"

    def __init__(self, path=None, option=None):

        if not path:
            path = "g++"

        super(GnuCppCppCompiler, self).__init__(path, option)

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
            opts = "-dynamiclib -fPIC " + super(GnuCppCppCompiler, self).get_option()

        elif sys.platform == "linux":
            opts = "-shared -fPIC " + super(GnuCppCppCompiler, self).get_option()

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts

class GnuFortranFortranCompiler(FortranFortranCompiler):

    vendor = "gnu"
    opt_openmp = "-fopenmp"
    opt_moddir = "-J %s"

    def __init__(self, path=None, option=None):

        if not path:
            path = "gfortran"

        super(GnuFortranFortranCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()
        
        if sys.platform in ("darwin", "linux"):
            if items[:3] == [b'GNU', b'Fortran', b'(GCC)']:
                return items[3].decode().split(".")
            raise Exception("Unknown compiler version syntax: %s" % str(items[:3]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "darwin":
            opts = "-dynamiclib -fPIC " + super(GnuFortranFortranCompiler, self).get_option()

        elif sys.platform == "linux":
            opts = "-shared -fPIC " + super(GnuFortranFortranCompiler, self).get_option()

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts


################
# Cray Compilers
################

class CrayClangCppCppCompiler(CppCppCompiler):

    vendor = "crayclang"
    opt_openmp = "--fopenmp"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("CC"):
            path = "CC"

        elif which("crayCC"):
            path = "crayCC"

        elif which("clang++"):
            path = "clang++"

        super(CrayClangCppCppCompiler, self).__init__(path, option)

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
            opts = "-shared -fPIC " + super(CrayClangCppCppCompiler, self).get_option()

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts

class CrayFortranFortranCompiler(FortranFortranCompiler):

    vendor = "cray"
    opt_openmp = "-h omp"
    opt_moddir = "-J %s"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("ftn"):
            path = "ftn"

        elif which("crayftn"):
            path = "crayftn"

        super(CrayFortranFortranCompiler, self).__init__(path, option)

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
            moddir = self.opt_moddir % self._blddir
            opts = ("-shared -fPIC %s " % moddir +
                    super(CrayFortranFortranCompiler, self).get_option())

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts

################
# AMD Compilers
################

class AmdClangCppCppCompiler(CppCppCompiler):

    vendor = "amdclang"
    opt_openmp = "-fopenmp"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("clang++"):
            path = "clang++"

        super(AmdClangCppCppCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()

        if sys.platform == "linux":
            if items[:2] == [b'clang', b'version']:
                return items[2].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:2]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "linux":
            opts = "-shared -fPIC " + super(AmdClangCppCppCompiler, self).get_option()

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts

class AmdFlangFortranFortranCompiler(FortranFortranCompiler):

    vendor = "amdflang"
    opt_openmp = "-fopenmp"
    opt_moddir = "-J %s"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("flang"):
            path = "flang"

        super(AmdFlangFortranFortranCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()
        
        if sys.platform == "linux":
            if items[:2] == [b'flang-new', b'version']:
                return items[2].decode().split(".")
            raise Exception("Unknown compiler version syntax: %s" % str(items[:2]))

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

    def get_option(self):

        if sys.platform == "linux":
            moddir = self.opt_moddir % self._blddir
            opts = ("-shared -fPIC %s " % moddir +
                    super(AmdFlangFortranFortranCompiler, self).get_option())

        else:
            raise Exception("Platform '%s' is not supported." % str(sys.platform))

        return opts

###################
# IBM XL Compilers
###################

class IbmXlCppCppCompiler(CppCppCompiler):

    vendor = "ibmxl"
    opt_openmp = "-qsmp=omp"
    opt_version = "-qversion"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("xlc++_r"):
            path = "xlc++_r"

        elif which("xlc++"):
            path = "xlc++"

        super(IbmXlCppCppCompiler, self).__init__(path, option)

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
            opts = "-shared -fPIC " + super(IbmXlCppCppCompiler, self).get_option()

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts

class IbmXlFortranFortranCompiler(FortranFortranCompiler):

    vendor = "ibmxl"
    opt_openmp = "-qsmp=omp"
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

        super(IbmXlFortranFortranCompiler, self).__init__(path, option)

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
            moddir = self.opt_moddir % self._blddir
            opts = ("-qmkshrobj -qpic %s " % moddir +
                    super(IbmXlFortranFortranCompiler, self).get_option())

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts


###################
# PGI Compilers
###################

class PgiCppCppCompiler(CppCppCompiler):

    vendor = "pgi"
    opt_openmp = "-mp"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("pgc++"):
            path = "pgc++"

        super(PgiCppCppCompiler, self).__init__(path, option)

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
            opts = "-shared -fpic " + super(PgiCppCppCompiler, self).get_option()

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts

class PgiFortranFortranCompiler(FortranFortranCompiler):

    vendor = "pgi"
    opt_openmp = "-mp"
    opt_moddir = "-module %s"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("pgfortran"):
            path = "pgfortran"

        super(PgiFortranFortranCompiler, self).__init__(path, option)

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
            moddir = self.opt_moddir % self._blddir
            opts = ("-shared -fpic %s " % moddir +
                    super(PgiFortranFortranCompiler, self).get_option())

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts


###################
# Intel Compilers
###################

class IntelCppCppCompiler(CppCppCompiler):

    vendor = "intel"
    opt_openmp = "-qopenmp"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("icpc"):
            path = "icpc"

        super(IntelCppCppCompiler, self).__init__(path, option)

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
            opts = "-shared -fpic " + super(IntelCppCppCompiler, self).get_option()

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts

class IntelFortranFortranCompiler(FortranFortranCompiler):

    vendor = "intel"
    opt_openmp = "-qopenmp"
    opt_moddir = "-module %s"

    def __init__(self, path=None, option=None):

        if path:
            pass

        elif which("ifort"):
            path = "ifort"

        super(IntelFortranFortranCompiler, self).__init__(path, option)

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
            moddir = self.opt_moddir % self._blddir
            opts = ("-shared -fpic %s " % moddir +
                    super(IntelFortranFortranCompiler, self).get_option())

        else:
            raise Exception("'%s' platform is not supported yet." % sys.platform)

        return opts


# priorities
_lang_priority = ["fortran", "cpp"]
_accel_priority = ["fortran", "cpp"]
_vendor_priority = ["intel", "pgi", "ibmxl", "amdflang", "amdclang", "crayclang", "cray", "gnu"]

def _langsort(l):
    return _lang_priority.index(l.lang)

def _accelsort(a):
    return _accel_priority.index(a.accel)

def _vendorsort(v):
    return _vendor_priority.index(v.vendor)

for langsubc in sorted(Compiler.__subclasses__(), key=_langsort):
    lang = langsubc.lang
    Compiler.avails[lang] = OrderedDict()

    for accelsubc in sorted(langsubc.__subclasses__(), key=_accelsort):
        accel = accelsubc.accel
        Compiler.avails[lang][accel] = OrderedDict()

        for vendorsubc in sorted(accelsubc.__subclasses__(), key=_vendorsort):
            vendor = vendorsubc.vendor
            Compiler.avails[lang][accel][vendor] = vendorsubc
