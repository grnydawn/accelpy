"""accelpy module"""


from .util import load_sharedlib, invoke_sharedlib
from .compile import build_sharedlib
from .accel import Accel
from .kernel import Kernel
from .fortran import OmptargetFortranAccel
