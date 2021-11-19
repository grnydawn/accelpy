"""accelpy C++ Accelerator module"""


from accelpy.accel import AccelBase
from accelpy.compiler import get_compilers

t_main = """
"""

class CppAccel(AccelBase):

    name = "cpp"
    lang = "cpp"

    def build_sharedlib(self, compilers=None):

        sharedlib = None

        self._order.update_argnames(self._inputs, self._outputs)

        fmt = {}
        code = t_main.format(**fmt)

        if compilers is None:
            compilers = get_compilers(self.name)

        for comp in compilers:
            try:
                lib = comp.compile(code)
                lib.testrun() 
                import pdb; pdb.set_trace()
                return lib
            except Exception as err:
                import pdb; pdb.set_trace()
                print(err)

        return sharedlib


AccelBase.avails[CppAccel.name] = CppAccel
