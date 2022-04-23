import numpy as np
from accelpy import Accel, Kernel

vecadd = "vecadd.knl"

N1 = 10
N2 = 20

x = np.ones((N1, N2), order="F", dtype=np.int64)
y = np.ones((N1, N2), order="F", dtype=np.int64)
z = np.zeros((N1, N2), order="F", dtype=np.int64)

acctarget = "openacc" # "openacc" # "omptarget"

accel = Accel(accel=acctarget, copyin=(x, y), copyout=(z,))

accel.launch(Kernel(vecadd), x, y, z)

accel.stop()

#import pdb; pdb.set_trace()

print("SUCCESS" if np.array_equal(z, x+y) else "FAIL")
