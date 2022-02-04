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
}


###################
#  Data
###################

def _gendata(testname):

    N1 = 100
    N3 = (2, 5, 10)

    data = []

    if testname == "vecadd1d":
        data.append(np.arange(N1, dtype=np.int64))
        data.append(np.arange(N1, dtype=np.int64) * 2)
        data.append(np.zeros(N1, dtype=np.int64))

    elif testname == "matmul":
        data.append(np.reshape(np.arange(100, dtype=np.float64), (4, 25), order="F"))
        data.append(np.reshape(np.arange(100, dtype=np.float64) * 2, (25, 4), order="F"))
        data.append(np.reshape(np.zeros(16, dtype=np.float64), (4, 4), order="F"))

    elif testname == "vecadd3d":
        data.append(np.reshape(np.arange(100, dtype=np.int64), N3, order="F"))
        data.append(np.reshape(np.arange(100, dtype=np.int64) * 2, N3, order="F"))
        data.append(np.reshape(np.zeros(100, dtype=np.int64), N3, order="F"))

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

def get_testdata(testname):
    return _gendata(testname), orders[testname]


def assert_testdata(testname, data):
    check_result(testname, data)
