"""accelpy Compiler module"""

import os, sys, abc, time, threading, inspect, hashlib
from numpy.ctypeslib import load_library
from numpy import ndarray
from collections import OrderedDict

from accelpy.core import Object
from accelpy.util import shellcmd, which
from accelpy import _config


class Compiler(Object):
    """Compiler Base Class"""

    avails = dict()
    libext = "so"

    def __init__(self, path, option=None):

        self.version = []

        if isinstance(path, str):
            self.set_version(path)
            self.path = path

        elif isintance(path, (list, tuple)):
            for p in path:
                try:
                    self.set_version(path)
                    self.path = path
                    break
                except:
                    pass

            assert self.path
        else:
            import pdb; pdb.set_trace()

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

    def compile(self, code):

        lib = None

        blddir = _config["session"]["workdir"]

        # TODO: change to hash
        text = (code+ self.lang + self.accel + self.vendor +
                    "".join(self.version))
        name = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

        codepath = os.path.join(blddir, name + "." + self.codeext)
        with open(codepath, "w") as f:
            f.write(code)

        libpath = os.path.join(blddir, name + "." + self.libext)

        compile_cmd = "{compiler} {option} -o {outfile} {infile}".format(
                compiler=self.path, option=self.get_option(),
                outfile=libpath, infile=codepath)

        import pdb; pdb.set_trace()
        out = shellcmd(compile_cmd)

        if out.returncode != 0:
            raise Exception("Compilation fails: %s" % out.stderr)

        if not os.path.isfile(libpath):
            raise Exception("Shared library is not created.")

        lib = load_library(name, blddir)

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

        else:
            import pdb; pdb.set_trace()

    def get_option(self):

        if sys.platform == "darwin":
            opts = "-dynamiclib -fPIC " + super(GnuCppCppCompiler, self).get_option()

        else:
            import pdb; pdb.set_trace()

        return opts

# TODO: should be abstract
class FortranFortranCompiler(FortranCompiler):

    accel = "fortran"
    opt_version = "--version"

class GnuFortranFortranCompiler(FortranFortranCompiler):

    vendor = "gnu"
    opt_openmp = "--fopenmp"

    def __init__(self, path=None, option=None):

        if not path:
            path = "gfortran"

        super(GnuFortranFortranCompiler, self).__init__(path, option)

    def parse_version(self, stdout):

        items = stdout.split()
        
        if sys.platform == "darwin":
            if items[:3] == [b'GNU', b'Fortran', b'(GCC)']:
                return items[3].decode().split(".")
            raise Exception("Unknown version syntaxt: %s" % str(items[:3]))

        else:
            import pdb; pdb.set_trace()

    def get_option(self):

        if sys.platform == "darwin":
            opts = "-dynamiclib -fPIC " + super(GnuFortranFortranCompiler, self).get_option()

        else:
            import pdb; pdb.set_trace()

        return opts


# TODO: apply priority

for langsubc in Compiler.__subclasses__():
    lang = langsubc.lang
    Compiler.avails[lang] = OrderedDict()

    for accelsubc in langsubc.__subclasses__():
        accel = accelsubc.accel
        Compiler.avails[lang][accel] = OrderedDict()

        for vendorsubc in accelsubc.__subclasses__():
            vendor = vendorsubc.vendor
            Compiler.avails[lang][accel][vendor] = vendorsubc

