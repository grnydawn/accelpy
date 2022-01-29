"""accelpy basic tests"""

import numpy as np
from accelpy import Accel, CppAccel, FortranAccel, HipAccel, Order



#######################
# Order definitions
#######################

order_vecadd1d = """

set_argnames("x", "y", "z")

attrspec = {
    'x': {
        'dimension': '1:'
    }
}

cpp_enable = True

[cpp: enable=cpp_enable]
    for (int id = 0; id < x.shape[0]; id++) {
        z(id) = x(id) + y(id);
    }

[fortran: a, b, c, attrspec=attrspec]
    INTEGER id

    DO id=LBOUND(a,1), UBOUND(a,1)
        c(id) = a(id) + b(id)
    END DO

[hip, cuda]
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < x.size) z(id) = x(id) + y(id);

[openacc_cpp]
    #pragma acc loop gang worker vector
    for (int id = 0; id < x.shape[0]; id++) {
        z(id) = x(id) + y(id);
    }

[openacc_fortran]
    INTEGER id

    !$acc loop gang worker vector
    DO id=1, x_attr%shape(1)
        z(id) = x(id) + y(id)
    END DO
    !$acc end loop

[openmp_cpp]

    #pragma omp for
    for (int id = 0; id < x.shape[0]; id++) {
        z(id) = x(id) + y(id);
        //printf("thread = %d\\n", omp_get_thread_num());
    }

[openmp_fortran]
    INTEGER id

    !$omp do
    DO id=1, x_attr%shape(1)
        z(id) = x(id) + y(id)
        !print *, omp_get_thread_num()
    END DO
    !$omp end do

"""

order_vecadd3d = """

set_argnames("x", "y", "z")

[cpp]
    for (int i = 0; i < x.shape[0]; i++) {
        for (int j = 0; j < x.shape[1]; j++) {
            for (int k = 0; k < x.shape[2]; k++) {
                z(i, j, k) = x(i, j, k) + y(i, j, k);
            }
        }
    }

[fortran]
    INTEGER i, j, k

    DO i=LBOUND(x, 1), UBOUND(x, 1)
        DO j=LBOUND(x, 2), UBOUND(x, 2)
            DO k=LBOUND(x, 3), UBOUND(x, 3)
                z(i, j, k) = x(i, j, k) + y(i, j, k)
            END DO
        END DO
    END DO

[hip, cuda]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < x.shape[0] && j < x.shape[1] && k < x.shape[2])
        z(i, j, k) = x(i, j, k) + y(i, j, k);

[openacc_cpp]

    #pragma acc loop gang
    for (int i = 0; i < x.shape[0]; i++) {
        #pragma acc loop worker
        for (int j = 0; j < x.shape[1]; j++) {
            #pragma acc loop vector
            for (int k = 0; k < x.shape[2]; k++) {
                z(i, j, k) = x(i, j, k) + y(i, j, k);
            }
        }
    }

[openacc_fortran]
    INTEGER i, j, k

    !$acc loop gang
    DO i=1, x_attr%shape(1)
        !$acc loop worker
        DO j=1, x_attr%shape(2)
            !$acc loop vector
            DO k=1, x_attr%shape(3)
                z(i, j, k) = x(i, j, k) + y(i, j, k)
            END DO
        END DO
    END DO

[openmp_cpp]
    #pragma omp for
    for (int i = 0; i < x.shape[0]; i++) {
        for (int j = 0; j < x.shape[1]; j++) {
            for (int k = 0; k < x.shape[2]; k++) {
                z(i, j, k) = x(i, j, k) + y(i, j, k);
            }
        }
    }

[openmp_fortran]
    INTEGER i, j, k

    !$omp do
    DO i=1, x_attr%shape(1)
        DO j=1, x_attr%shape(2)
            DO k=1, x_attr%shape(3)
                z(i, j, k) = x(i, j, k) + y(i, j, k)
            END DO
        END DO
    END DO
    !$omp end do

"""

order_matmul = """

set_argnames("X", "Y", "Z")

[cpp]

    for (int i = 0; i < X.shape[0]; i++) {
        for (int j = 0; j < Y.shape[1]; j++) {
            Z(i, j) = 0.0;
            for (int k = 0; k < Y.shape[0]; k++) {
                Z(i, j) += X(i, k) * Y(k, j);
            }
        }
    }

[fortran]
    INTEGER i, j, k

    DO i=LBOUND(X, 1), UBOUND(X, 1)
        DO j=LBOUND(Y, 2), UBOUND(Y, 2)
            Z(i, j) = 0
            DO k=LBOUND(Y, 1), UBOUND(Y, 1)
                Z(i, j) = Z(i, j) + X(i, k) * Y(k, j)
            END DO
        END DO
    END DO

[hip, cuda]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < X.shape[0] && j < Y.shape[1]) {
        Z(i, j) = 0.0;
        for (int k = 0; k < Y.shape[0]; k++) {
            Z(i, j) += X(i, k) * Y(k, j);
        }
    }

[openacc_cpp]

    #pragma acc loop gang
    for (int i = 0; i < X.shape[0]; i++) {
        #pragma acc loop worker
        for (int j = 0; j < Y.shape[1]; j++) {
            Z(i, j) = 0.0;
            for (int k = 0; k < Y.shape[0]; k++) {
                Z(i, j) += X(i, k) * Y(k, j);
            }
        }
    }

[openacc_fortran]
    INTEGER i, j, k

    !$acc loop gang
    DO i=1, X_attr%shape(1)
        !$acc loop worker
        DO j=1, Y_attr%shape(2)
            Z(i, j) = 0
            DO k=1, Y_attr%shape(1)
                Z(i, j) = Z(i, j) + X(i, k) * Y(k, j)
            END DO
        END DO
    END DO

[openmp_cpp]

    #pragma omp for
    for (int i = 0; i < X.shape[0]; i++) {
        for (int j = 0; j < Y.shape[1]; j++) {
            Z(i, j) = 0.0;
            for (int k = 0; k < Y.shape[0]; k++) {
                Z(i, j) += X(i, k) * Y(k, j);
            }
        }
    }

[openmp_fortran]

    INTEGER i, j, k

    !$omp do
    DO i=1, X_attr%shape(1)
        DO j=1, Y_attr%shape(2)
            Z(i, j) = 0
            DO k=1, Y_attr%shape(1)
                Z(i, j) = Z(i, j) + X(i, k) * Y(k, j)
            END DO
        END DO
    END DO
    !$omp end do

"""

#######################
# Tests
#######################

N1 = 100
a_1d = np.arange(N1, dtype=np.int64)
b_1d = np.arange(N1, dtype=np.int64) * 2
c_1d = np.zeros(N1, dtype=np.int64)

N3 = (2, 5, 10)
a_3d = np.reshape(np.arange(100, dtype=np.int64), N3, order="F")
b_3d = np.reshape(np.arange(100, dtype=np.int64) * 2, N3, order="F")
c_3d = np.reshape(np.zeros(100, dtype=np.int64), N3, order="F")

a_2d = np.reshape(np.arange(100, dtype=np.float64), (4, 25), order="F")
b_2d = np.reshape(np.arange(100, dtype=np.float64) * 2, (25, 4), order="F")
c_2d = np.reshape(np.zeros(16, dtype=np.float64), (4, 4), order="F")

def main():

    c_2d.fill(0)

    accel = Accel(a_2d, b_2d, Order(order_matmul), c_2d,
                    kind=["fortran"], compile=["gnu"])

    accel.run(c_2d.shape)

    accel.stop()

    assert np.array_equal(c_2d, np.matmul(a_2d, b_2d))

if __name__ == "__main__":
    main()
    import pdb; pdb.set_trace()
