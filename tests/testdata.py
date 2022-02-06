"test configuration and data"

import numpy as np

###################
#  Orders
###################

orders = {
    "vecadd1d": """

set_argnames("x", "y", "z")
attrspec = { 'x': { 'dimension': '1:' } }

cpp_enable = True

[cpp: enable=cpp_enable]

    //for (int id = 0; id < shape_x[0]; id++) {
    for (int id = 0; id < SHAPE(x,0); id++) {
        z[id] = x[id] + y[id];
    }

[fortran: a, b, c, attrspec=attrspec]
    INTEGER id, begin, end

    begin = LBOUND(a,1) 
    end = UBOUND(a,1) 

    DO id=begin, end
        c(id) = a(id) + b(id)
    END DO

[hip: kernel="hip_kernel"]


    hipMalloc((void **)&DPTR(x), SIZE(x) * sizeof(TYPE(x)));
    hipMemcpyHtoD(DVAR(x), VAR(x), SIZE(x) * sizeof(TYPE(x)));

    hipMalloc((void **)&DPTR(y), SIZE(y) * sizeof(TYPE(y)));
    hipMemcpyHtoD(DVAR(y), VAR(y), SIZE(y) * sizeof(TYPE(y)));

    hipMalloc((void **)&DPTR(z), SIZE(z) * sizeof(TYPE(z)));
    //accelpy_kernel<<<1, SIZE(x)>>>(DVAR(x), DVAR(y), DVAR(z));
    accelpy_kernel<<<1, SHAPE(x,0) >>>(DVAR(x), DVAR(y), DVAR(z));

    hipMemcpyDtoH(VAR(z), DVAR(z), SIZE(z) * sizeof(TYPE(z)));

    hipFree(DPTR(x));
    hipFree(DPTR(y));
    hipFree(DPTR(z));
/*
*/


[hip_kernel]

    __global__ void accelpy_kernel(ARG(x), ARG(y), ARG(z)){

        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if(id < SHAPE(x, 0)) z[id] = x[id] + y[id];
        //if(id < SIZE(x)) z[id] = x[id] + y[id];

    }


[openacc_cpp]
    int length = shape_x[0];

    #pragma acc data copyin(x[0:length], y[0:length]) copyout(z[0:length])
    {
    #pragma acc parallel loop worker 
    for (int id = 0; id < length; id++) {
        z[id] = x[id] + y[id];
    }
    }

[openacc_fortran]
    INTEGER id, begin, end

    begin = LBOUND(x,1) 
    end = UBOUND(x,1) 

    !$acc data copyin(x(begin:end), y(begin:end)) copyout(z(begin:end))
    !$acc parallel num_workers(end-begin+1) 
    !$acc loop worker
    DO id=begin, end
        z(id) = x(id) + y(id)
    END DO
    !$acc end parallel
    !$acc end data

[openmp_cpp]

    #pragma omp for
    for (int id = 0; id < shape_x[0]; id++) {
        z[id] = x[id] + y[id];
        //printf("thread = %d\\n", omp_get_thread_num());
    }

[openmp_fortran]
    INTEGER id, begin, end 

    begin = LBOUND(x,1) 
    end = UBOUND(x,1) 

    !$omp do
    DO id=begin, end
        z(id) = x(id) + y(id)
        !print *, omp_get_thread_num()
    END DO
    !$omp end do

""",

    "vecadd3d": """
set_argnames("x", "y", "z")

[cpp]
    for (int i = 0; i < shape_x[0]; i++) {
        for (int j = 0; j < shape_x[1]; j++) {
            for (int k = 0; k < shape_x[2]; k++) {
                z[i][j][k] = x[i][j][k] + y[i][j][k];
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

    int len0 = shape_x[0];
    int len1 = shape_x[1];
    int len2 = shape_x[2];

    #pragma acc data copyin(x[0:len0][0:len1][0:len2], y[0:len0][0:len1][0:len2]) copyout(z[0:len0][0:len1][0:len2])
    {
    #pragma acc parallel
    {
    #pragma acc loop gang
    for (int i = 0; i < len0; i++) {
        #pragma acc loop worker
        for (int j = 0; j < len1; j++) {
            #pragma acc loop vector
            for (int k = 0; k < len2; k++) {
                z[i][j][k] = x[i][j][k] + y[i][j][k];
            }
        }
    }
    }
    }

[openacc_fortran]
    INTEGER i, j, k, b1, b2, b3, e1, e2, e3

    b1 = LBOUND(x,1) 
    b2 = LBOUND(x,2) 
    b3 = LBOUND(x,3) 
    e1 = UBOUND(x,1) 
    e2 = UBOUND(x,2) 
    e3 = UBOUND(x,3) 

    !$acc data copyin(x(b1:e1, b2:e2, b3:e3), y(b1:e1, b2:e2, b3:e3)), copyout(z(b1:e1, b2:e2, b3:e3))
    !$acc parallel num_gangs(e1-b1+1) num_workers(e2-b2+1) vector_length(e3-b3+1)
    !$acc loop gang
    DO i=b1, e1
        !$acc loop worker
        DO j=b2, e2
            !$acc loop vector
            DO k=b3, e3
                z(i, j, k) = x(i, j, k) + y(i, j, k)
            END DO
        END DO
    END DO
    !$acc end parallel
    !$acc end data

[openmp_cpp]
    #pragma omp for
    for (int i = 0; i < shape_x[0]; i++) {
        for (int j = 0; j < shape_x[1]; j++) {
            for (int k = 0; k < shape_x[2]; k++) {
                z[i][j][k] = x[i][j][k] + y[i][j][k];
            }
        }
    }

[openmp_fortran]
    INTEGER i, j, k, b1, b2, b3, e1, e2, e3

    b1 = LBOUND(x,1) 
    b2 = LBOUND(x,2) 
    b3 = LBOUND(x,3) 
    e1 = UBOUND(x,1) 
    e2 = UBOUND(x,2) 
    e3 = UBOUND(x,3) 

    !$omp do
    DO i=b1, e1
        DO j=b2, e2
            DO k=b3, e3
                z(i, j, k) = x(i, j, k) + y(i, j, k)
            END DO
        END DO
    END DO
    !$omp end do

""",

    "matmul": """

set_argnames("X", "Y", "Z")

[cpp]

    for (int i = 0; i < shape_X[0]; i++) {
        for (int j = 0; j < shape_Y[1]; j++) {
            Z[i][j] = 0.0;
            for (int k = 0; k < shape_Y[0]; k++) {
                Z[i][j] = Z[i][j] + X[i][k] * Y[k][j];
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

    int x0 = shape_X[0];
    int x1 = shape_X[1];
    int y0 = shape_Y[0];
    int y1 = shape_Y[1];
    int z0 = shape_Z[0];
    int z1 = shape_Z[1];

    #pragma acc data copyin(X[0:x0][0:x1], Y[0:y0][0:y1]) copyout(Z[0:z0][0:z1])
    {
    #pragma acc parallel
    {
    #pragma acc loop gang
    for (int i = 0; i < x0; i++) {
        #pragma acc loop worker
        for (int j = 0; j < y1; j++) {
            Z[i][j] = 0.0;
            for (int k = 0; k < y0; k++) {
                Z[i][j] = Z[i][j] + X[i][k] * Y[k][j];
            }
        }
    }
    }
    }

[openacc_fortran]
    INTEGER i, j, k, xl1, xu1, xl2, xu2, yl1, yu1, yl2, yu2, zl1, zu1, zl2, zu2

    xl1 = LBOUND(X,1) 
    xu1 = UBOUND(X,1) 
    xl2 = LBOUND(X,2) 
    xu2 = UBOUND(X,2) 
    yl1 = LBOUND(Y,1) 
    yu1 = UBOUND(Y,1) 
    yl2 = LBOUND(Y,2) 
    yu2 = UBOUND(Y,2) 
    zl1 = LBOUND(Z,1) 
    zu1 = UBOUND(Z,1) 
    zl2 = LBOUND(Z,2) 
    zu2 = UBOUND(Z,2) 

    !$acc data copyin(X(xl1:xu1, xl2:xu2), Y(yl1:yu1, yl2:yu2)), copyout(Z(zl1:zu1, zl2:zu2))
    !$acc parallel num_gangs(xu1-xl1+1) num_workers(yu2-yl2+1)
    !$acc loop gang
    DO i=xl1, xu1
        !$acc loop worker
        DO j=yl2, yu2
            Z(i, j) = 0
            DO k=yl1, yu1
                Z(i, j) = Z(i, j) + X(i, k) * Y(k, j)
            END DO
        END DO
    END DO
    !$acc end parallel
    !$acc end data

[openmp_cpp]

    #pragma omp for
    for (int i = 0; i < shape_X[0]; i++) {
        for (int j = 0; j < shape_Y[1]; j++) {
            Z[i][j] = 0.0;
            for (int k = 0; k < shape_Y[0]; k++) {
                Z[i][j] = Z[i][j] + X[i][k] * Y[k][j];
            }
        }
    }

[openmp_fortran]

    INTEGER i, j, k, xl1, xu1, yl1, yu1, yl2, yu2

    xl1 = LBOUND(X,1) 
    xu1 = UBOUND(X,1) 
    yl1 = LBOUND(Y,1) 
    yu1 = UBOUND(Y,1) 
    yl2 = LBOUND(Y,2) 
    yu2 = UBOUND(Y,2) 

    !$omp do
    DO i=xl1, xu1
        DO j=yl2, yu2
            Z(i, j) = 0
            DO k=yl1, yu1
                Z(i, j) = Z(i, j) + X(i, k) * Y(k, j)
            END DO
        END DO
    END DO
    !$omp end do

"""
}


###################
#  Data
###################

def _gendata(testname, lang):

    N1 = 100
    N3 = (2, 5, 10)

    if lang == "cpp":
        order = "C"

    elif lang == "fortran":
        order = "F"

    data = []

    if testname == "vecadd1d":
        data.append(np.arange(N1, dtype=np.int64))
        data.append(np.arange(N1, dtype=np.int64) * 2)
        data.append(np.zeros(N1, dtype=np.int64))

    elif testname == "matmul":
        data.append(np.reshape(np.arange(100, dtype=np.float64), (4, 25), order=order))
        data.append(np.reshape(np.arange(100, dtype=np.float64) * 2, (25, 4), order=order))
        data.append(np.reshape(np.zeros(16, dtype=np.float64), (4, 4), order=order))

    elif testname == "vecadd3d":
        data.append(np.reshape(np.arange(100, dtype=np.int64), N3, order=order))
        data.append(np.reshape(np.arange(100, dtype=np.int64) * 2, N3, order=order))
        data.append(np.reshape(np.zeros(100, dtype=np.int64), N3, order=order))

    else:
        assert False

    return data


def check_result(testname, data):

    if testname == "vecadd1d":
        assert np.array_equal(data[2], data[0] + data[1])

    elif testname == "matmul":
        assert np.array_equal(data[2], np.matmul(data[0], data[1]))

    elif testname == "vecadd3d":
        assert np.array_equal(data[2], data[0] + data[1])

    else:
        assert False


###################
#  Functions
###################

def get_testdata(testname, lang):
    return _gendata(testname, lang), orders[testname]


def assert_testdata(testname, data):
    check_result(testname, data)
