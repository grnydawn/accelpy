"""accelpy basic tests"""

import numpy as np
import pytest
from accelpy import Accel, CppAccel, FortranAccel


test_accels = ("cpp", "fortran")

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

order_vecadd3d = """

set_argnames(("x", "y"), "z")

[cpp]
    for (int i = 0; i < x.shape(0); i++) {
        for (int j = 0; j < x.shape(1); j++) {
            for (int k = 0; k < x.shape(2); k++) {
                z(i, j, k) = x(i, j, k) + y(i, j, k);
            }
        }
    }

[fortran]
    INTEGER i, j, k

    DO i=1, x_attr%shape(1)
        DO j=1, x_attr%shape(2)
            DO k=1, x_attr%shape(3)
                z(i, j, k) = x(i, j, k) + y(i, j, k)
            END DO
        END DO
    END DO
"""

order_matmul = """

set_argnames(("X", "Y"), "Z")

[cpp]

    for (int i = 0; i < X.shape(0); i++) {
        for (int j = 0; j < Y.shape(1); j++) {
            Z(i, j) = 0.0;
            for (int k = 0; k < Y.shape(0); k++) {
                Z(i, j) += X(i, k) * Y(k, j);
            }
        }
    }

[fortran]
    INTEGER i, j, k

    DO i=1, X_attr%shape(1)
        DO j=1, Y_attr%shape(2)
            Z(i, j) = 0
            DO k=1, Y_attr%shape(1)
                Z(i, j) = Z(i, j) + X(i, k) * Y(k, j)
            END DO
        END DO
    END DO
"""

#######################
# Tests
#######################

N1 = 100
a_1d = np.arange(N1, dtype=np.int64)
b_1d = np.arange(N1, dtype=np.int64) * 2
c_1d = np.zeros(N1, dtype=np.int64)

N3 = (2, 5, 10)
a_3d = np.reshape(np.arange(N1, dtype=np.int64), N3)
b_3d = np.reshape(np.arange(N1, dtype=np.int64) * 2, N3)
c_3d = np.reshape(np.zeros(N1, dtype=np.int64), N3)


a_2d = np.reshape(np.arange(N1, dtype=np.float64), (4, 25))
b_2d = np.reshape(np.arange(N1, dtype=np.float64) * 2, (25, 4))
c_2d = np.reshape(np.zeros(16, dtype=np.float64), (4, 4))

@pytest.mark.parametrize("accel", test_accels)
def test_first(accel):

    c_1d.fill(0)

    accel_cpp = CppAccel(order_vecadd1d, (a_1d, b_1d), c_1d)

    accel_cpp.run()

    accel_cpp.wait()

    assert np.array_equal(c_1d, a_1d + b_1d)

    c_1d.fill(0)

    accel_fortran = FortranAccel(order_vecadd1d, (a_1d, b_1d), c_1d)

    accel_fortran.run()

    accel_fortran.wait()

    assert np.array_equal(c_1d, a_1d + b_1d)

@pytest.mark.parametrize("accel", test_accels)
def test_add3d(accel):

    c_3d.fill(0)

    accel = Accel(order_vecadd3d, (a_3d, b_3d), c_3d, kind=[accel])

    accel.run()

    accel.wait()

    assert np.array_equal(c_3d, a_3d + b_3d)

@pytest.mark.parametrize("accel", test_accels)
def test_matmul(accel):

    c_2d.fill(0)

    accel = Accel(order_matmul, (a_2d, b_2d), c_2d, kind=[accel])

    accel.run()

    accel.wait()

    assert np.array_equal(c_2d, np.matmul(a_2d, b_2d))

