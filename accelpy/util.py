"""accelpy utility functions"""

from subprocess import PIPE, run as subp_run
from ctypes import c_int, CDLL, RTLD_GLOBAL
from numpy.ctypeslib import ndpointer


class System:
    name = "cray"


def shellcmd(cmd, shell=True, stdout=PIPE, stderr=PIPE,
             check=False):

    return subp_run(cmd, shell=shell, stdout=stdout,
                    stderr=stderr, check=check)


def load_sharedlib(libpath):

    return CDLL(libpath, mode=RTLD_GLOBAL)

def invoke_sharedlib(libobj, funcname, *args):

    func = getattr(libobj, funcname)
    func.restype = c_int
    func.argtypes = [ndpointer(a.dtype) for a in args]

    return func(*args)

def get_system():
    return System()

