"""accelpy Compiler module"""

import os, abc, time, threading, inspect
from numpy.ctypeslib import load_library
from numpy import ndarray
from collections import OrderedDict

from accelpy.core import Object
from accelpy.accel import AccelBase
from accelpy.util import shellcmd
from accelpy import _config


class Compiler(Object):
    """Compiler Base Class"""

    avails = dict()

    def __init__(self, option=None):
        self.option = option

    @abc.abstractmethod
    def get_path(self):
        pass

    @abc.abstractmethod
    def get_option(self):
        pass

    def compile(self, code):
        lib = None

        codepath = os.path.join(self.workdir, fname + "." + self.codeext)
        with open(codepath, "w") as f:
            f.write(code)

        cmd = "{compiler} {option} -o {outfile} {infile}".format(
                compiler=self.get_path(), option=self.get_option(),
                outfile=outfile, infile=infile)

        out = shellcmd(cmd)

        return lib


# TODO: should be abstract
class CppCompiler(Compiler):

    lang = "cpp"

# TODO: should be abstract
class FortranCompiler(Compiler):

    lang = "fortran"

# TODO: should be abstract
class CppCppCompiler(CppCompiler):

    accel = "cpp"

# TODO: should be abstract
class GnuCppCppCompiler(CppCppCompiler):

    vendor = "gnu"

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


def generate_compilor(compile):

    clist = compile.split()

    compcls = Compiler.from_path(clist[0])
    comp = compcls(option=" ".join(clist[1:]))

    return comp


def get_compilers(accel, compiler=None):
    """
        parameters:

        accel: the accelerator id or a list of them
        compiler: compiler path or compiler id, or a list of them 
                  syntax: "vendor|path [{default}] [additional options]"
"""

    accels = []

    if isinstance(accel, str):
        accels.append((accel, AccelBase.avails[accel].lang))

    elif isinstance(accel, AccelBase):
        accels.append((accel.name, accel.lang))

    elif isinstance(accel, (list, tuple)):
        for a in accel:
            if isinstance(a, str):
                accels.append((a, AccelBase.avails[a].lang))

            elif isinstance(a, AccelBase):
                accels.append((a.name, a.lang))

            else:
                raise Exception("Unknown accelerator type: %s" % str(a))

    else:
        raise Exception("Unknown accelerator type: %s" % str(accel))

    compilers = []

    if compiler:
        if isinstance(compiler, str):
            citems = compiler.split()

            if not citems:
                raise Exception("Blank compiler")

            if os.path.isfile(citems[0]):
                compilers.append(generate_compiler(compiler))

            else:
                # TODO: vendor name search
                for lang, langsubc in Compiler.avalis.items():
                    for accel, accelsubc in langsubc.items():
                        for vendor, vendorsubc in accelsubc.items():
                            if vendor == citems[0]:
                                try:
                                    compilers.append(vendorsubc(option=" ".join(citems[1:])))
                                except:
                                    pass

        elif isinstance(compiler, Compiler):
            compilers.append(compiler)

        elif isinstance(compiler (list, tuple)):

            for comp in compiler:
                if isinstance(comp, str):
                    citems = comp.split()

                    if not citems:
                        raise Exception("Blank compilor")

                    if os.path.isfile(citems[0]):
                        try:
                            compilers.append(generate_compilor(comp))
                        except:
                            pass

                    else:
                        # TODO: vendor name search
                        for lang, langsubc in Compiler.avalis.items():
                            for accel, accelsubc in langsubc.items():
                                for vendor, vendorsubc in accelsubc.items():
                                    if vendor == citems[0]:
                                        try:
                                            compilers.append(vendorsubc(option=" ".join(citems[1:])))
                                        except:
                                            pass

                elif isinstance(comp, Compiler):
                    compilers.append(comp)

                else:
                    raise Exception("Unsupported compiler type: %s" % str(comp))
        else:
            raise Exception("Unsupported compiler type: %s" % str(compiler))

    new_compilers = []

    if compilers:
        for comp in compiers:
            if any(comp.accel==a[0] and comp.lang==a[1] for a in accels):
                new_compilers.append(comp)
    else:
        for acc, lang in accels:
            vendors = Compiler.avails[lang][acc]
            for vendor, cls in vendors.items():
                try:
                    new_compilers.append(cls())
                except:
                    pass

    return new_compilers
