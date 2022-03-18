"""accelpy module"""


from .util import load_sharedlib, invoke_sharedlib
from .compile import build_sharedlib
from .kernel import Kernel
from .spec import Spec
from .fortran import OmptargetFortranKernel
