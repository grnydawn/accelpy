import numpy as np
from accelpy import Accel, Kernel

vecadd = "vecadd.knl"

N1 = 10
N2 = 20

x = np.ones((N1, N2), order="F")
y = np.ones((N1, N2), order="F")
z = np.zeros((N1, N2), order="F")

accel = Accel(copyin=(x, y), copyout=(z,))

accel.launch(Kernel(vecadd), x, y, z)

accel.stop()

print("SUCCESS" if np.array_equal(z, x+y) else "FAIL")
