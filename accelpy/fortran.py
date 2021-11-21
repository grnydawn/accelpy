"""accelpy Fortran Accelerator module"""


from accelpy.accel import AccelBase, get_compilers
from ctypes import c_int, c_longlong, c_float, c_double

t_main = """
"""

t_module = """
"""


class FortranAccel(AccelBase):

    name = "fortran"
    lang = "fortran"

    def gen_code(self, inputs, outputs):

        main_fmt = {
        }

        code = t_main.format(**main_fmt)

        return code

    def getname_h2amalloc(self, arg):
        return "accelpy_h2amalloc_%s" % arg["curname"]

    def getname_h2acopy(self, arg):
        return "accelpy_h2acopy_%s" % arg["curname"]

    def getname_a2hcopy(self, arg):
        return "accelpy_a2hcopy_%s" % arg["curname"]


AccelBase.avails[FortranAccel.name] = FortranAccel
