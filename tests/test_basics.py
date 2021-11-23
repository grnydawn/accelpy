"""accelpy basic tests"""

import numpy as np
from accelpy import Accel, CppAccel

#######################
# Order definitions
#######################

order_vecadd1d = """

set_argnames(("x", "y"), "z")

cpp_enable = True

[cpp: enable=cpp_enable]
    for (int id = 0; id < x.shape(0); id++) {
        z(id) = x(id) + y(id);
    }

[fortran]
    INTEGER id

    DO id=1, x_attr%shape(1)
        z(id) = x(id) + y(id)
    END DO

"""


#######################
# Tests
#######################

N1 = 100

a_1d = np.arange(N1, dtype=np.int64)
b_1d = np.arange(N1, dtype=np.int64) * 2
c_1d = np.zeros(N1, dtype=np.int64)


def test_first():

    c_1d.fill(0)

    accel = CppAccel(order_vecadd1d, (a_1d, b_1d), c_1d)

    accel.run()

    accel.wait()

    assert all(c_1d == a_1d + b_1d)


def test_multiaccel():

    c_1d.fill(0)

    accel = Accel(order_vecadd1d, (a_1d, b_1d), c_1d, kind=["cpp", "fortran"])
    #accel = Accel(order_vecadd1d, (a_1d, b_1d), c_1d, kind=["fortran", "cpp"])

    accel.run()

    accel.wait()

    #import pdb; pdb.set_trace()
    assert all(c_1d == a_1d + b_1d)
