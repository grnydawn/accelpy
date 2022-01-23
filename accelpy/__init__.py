"""accelpy package"""

# accelerators
from .accel import Accel, AccelBase

# order
from .order import Order 

# import accelerator per their priority
from .fortran import FortranAccel
from .openacc_fortran import OpenaccFortranAccel
from .openmp_fortran import OpenmpFortranAccel
from .cpp import CppAccel
from .hip import HipAccel
from .cuda import CudaAccel
from .openacc_cpp import OpenaccCppAccel
from .openmp_cpp import OpenmpCppAccel
