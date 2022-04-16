"test configuration and data"

import numpy as np

###################
#  Orders
###################

specs = {
    "vecadd1d": """

set_argnames("x", "y", "z")
attrspec = { 'x': { 'dimension': '1:' } }

cpp_enable = True

[cpp: enable=cpp_enable]

    //for (int id = 0; id < shape_x[0]; id++) {
    for (int id = 0; id < SHAPE(x,0); id++) {
        z[id] = x[id] + y[id];
    }

[fortran:a, b, c, attrspec=attrspec]
    INTEGER id, begin, end

    begin = LBOUND(a,1) 
    end = UBOUND(a,1) 

    DO id=begin, end
        c(id) = a(id) + b(id)
    END DO

[cuda, hip: gridsize=GRID, blocksize=BLOCK]

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < SHAPE(x, 0)) z[id] = x[id] + y[id];


[openacc_cpp]
    int length = shape_x[0];

    #pragma acc parallel loop worker 
    for (int id = 0; id < length; id++) {
        z[id] = x[id] + y[id];
    }

[openacc_fortran]
    INTEGER id, begin, end

    begin = LBOUND(x,1) 
    end = UBOUND(x,1) 

    !$acc parallel num_workers(end-begin+1) 
    !$acc loop worker
    DO id=begin, end
        z(id) = x(id) + y(id)
    END DO
    !$acc end parallel

[openmp_cpp]

    #pragma omp parallel for
    for (int id = 0; id < SHAPE(x, 0); id++) {
        z[id] = x[id] + y[id];
        printf("thread = %d\\n", omp_get_thread_num());
    }

[omptarget_cpp]

    #pragma omp target teams distribute parallel for
    for (int id = 0; id < SHAPE(x, 0); id++) {
        z[id] = x[id] + y[id];
        //printf("thread = %d\\n", omp_get_thread_num());
    }

[openmp_fortran]
    INTEGER id, begin, end 

    begin = LBOUND(x,1) 
    end = UBOUND(x,1) 

    !$omp parallel do
    DO id=begin, end
        z(id) = x(id) + y(id)
        print *, omp_get_thread_num()
    END DO
    !$omp end parallel do

[omptarget_fortran: x, y, z]
    INTEGER id, begin, end 

    begin = LBOUND(x,1) 
    end = UBOUND(x,1) 

    !$omp target
    !$omp parallel do 
    DO id=begin, end
        z(id) = x(id) + y(id)
        !print *, omp_get_thread_num()
    END DO
    !$omp end target

""",

    "vecadd3d": """

set_argnames("x", "y", "z")

[cpp]
    for (int i = 0; i < SHAPE(x, 0); i++) {
        for (int j = 0; j < SHAPE(x, 1); j++) {
            for (int k = 0; k < SHAPE(x,2); k++) {
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

[cuda, hip: gridsize=GRID, blocksize=BLOCK]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < SHAPE(x, 0) && j < SHAPE(x, 1) && k < SHAPE(x, 2))
        z[i][j][k] = x[i][j][k] + y[i][j][k];

[openacc_cpp]

    int len0 = SHAPE(x, 0);
    int len1 = SHAPE(x, 1);
    int len2 = SHAPE(x, 2);

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

[openacc_fortran]
    INTEGER i, j, k, b1, b2, b3, e1, e2, e3

    b1 = LBOUND(x,1) 
    b2 = LBOUND(x,2) 
    b3 = LBOUND(x,3) 
    e1 = UBOUND(x,1) 
    e2 = UBOUND(x,2) 
    e3 = UBOUND(x,3) 

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

[openmp_cpp]
    #pragma omp parallel for
    for (int i = 0; i < SHAPE(x, 0); i++) {
        for (int j = 0; j < SHAPE(x, 1); j++) {
            for (int k = 0; k < SHAPE(x, 2); k++) {
                z[i][j][k] = x[i][j][k] + y[i][j][k];
            }
        }
    }

[omptarget_cpp]

    #pragma omp target teams num_teams(SHAPE(x, 0))
    #pragma omp distribute
    for (int i = 0; i < SHAPE(x, 0); i++) {
        #pragma omp parallel for
        for (int j = 0; j < SHAPE(x, 1); j++) {
            for (int k = 0; k < SHAPE(x, 2); k++) {
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

    !$omp parallel do
    DO i=b1, e1
        DO j=b2, e2
            DO k=b3, e3
                z(i, j, k) = x(i, j, k) + y(i, j, k)
            END DO
        END DO
    END DO
    !$omp end parallel do

[omptarget_fortran: a, b, c]
    INTEGER i, j, k, b1, b2, b3, e1, e2, e3

    b1 = LBOUND(a,1) 
    b2 = LBOUND(a,2) 
    b3 = LBOUND(a,3) 
    e1 = UBOUND(a,1) 
    e2 = UBOUND(a,2) 
    e3 = UBOUND(a,3) 

    !$omp target
    !$omp teams num_teams(e1-b1+1)
    !$omp distribute
    DO i=b1, e1
        !$omp parallel do 
        DO j=b2, e2
            DO k=b3, e3
                c(i, j, k) = a(i, j, k) + b(i, j, k)
            END DO
        END DO
    END DO
    !$omp end teams
    !$omp end target

    !$omp target update from(c)
""",

    "matmul": """

set_argnames("A", "B", "C")

[cpp: X, Y, Z]

    for (int i = 0; i < SHAPE(X, 0); i++) {
        for (int j = 0; j < SHAPE(Y, 1); j++) {
            Z[i][j] = 0.0;
            for (int k = 0; k < SHAPE(Y, 0); k++) {
                Z[i][j] = Z[i][j] + X[i][k] * Y[k][j];
            }
        }
    }

[fortran: X, Y, Z]
    INTEGER i, j, k

    DO i=LBOUND(X, 1), UBOUND(X, 1)
        DO j=LBOUND(Y, 2), UBOUND(Y, 2)
            Z(i, j) = 0
            DO k=LBOUND(Y, 1), UBOUND(Y, 1)
                Z(i, j) = Z(i, j) + X(i, k) * Y(k, j)
            END DO
        END DO
    END DO

[cuda, hip: X, Y, Z, gridsize=GRID, blocksize=BLOCK]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < SHAPE(X, 0) && j < SHAPE(Y, 1)) {
        Z[i][j] = 0.0;
        for (int k = 0; k < SHAPE(Y, 0); k++) {
            Z[i][j] += X[i][k] * Y[k][j];
        }
    }

[openacc_cpp: X, Y, Z]

    int x0 = SHAPE(X, 0);
    int x1 = SHAPE(X, 1);
    int y0 = SHAPE(Y, 0);
    int y1 = SHAPE(Y, 1);
    int z0 = SHAPE(Z, 0);
    int z1 = SHAPE(Z, 1);

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

[openacc_fortran: X, Y, Z]
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

[openmp_cpp: X, Y, Z]

    #pragma omp parallel for
    for (int i = 0; i < SHAPE(X, 0); i++) {
        for (int j = 0; j < SHAPE(Y, 1); j++) {
            Z[i][j] = 0.0;
            for (int k = 0; k < SHAPE(Y, 0); k++) {
                Z[i][j] = Z[i][j] + X[i][k] * Y[k][j];
            }
        }
    }

[omptarget_cpp: X, Y, Z]

    #pragma omp target teams num_teams(SHAPE(X, 0))
    #pragma omp distribute
    for (int i = 0; i < SHAPE(X, 0); i++) {
        #pragma omp parallel for
        for (int j = 0; j < SHAPE(Y, 1); j++) {
            Z[i][j] = 0.0;
            for (int k = 0; k < SHAPE(Y, 0); k++) {
                Z[i][j] = Z[i][j] + X[i][k] * Y[k][j];
            }
        }
    }

[openmp_fortran: X, Y, Z]

    INTEGER i, j, k, xl1, xu1, yl1, yu1, yl2, yu2

    xl1 = LBOUND(X,1) 
    xu1 = UBOUND(X,1) 
    yl1 = LBOUND(Y,1) 
    yu1 = UBOUND(Y,1) 
    yl2 = LBOUND(Y,2) 
    yu2 = UBOUND(Y,2) 

    !$omp parallel do
    DO i=xl1, xu1
        DO j=yl2, yu2
            Z(i, j) = 0
            DO k=yl1, yu1
                Z(i, j) = Z(i, j) + X(i, k) * Y(k, j)
            END DO
        END DO
    END DO
    !$omp end parallel do

[omptarget_fortran: A, B, C]

    INTEGER i, j, k, xl1, xu1, yl1, yu1, yl2, yu2

    xl1 = LBOUND(A,1) 
    xu1 = UBOUND(A,1) 
    yl1 = LBOUND(B,1) 
    yu1 = UBOUND(B,1) 
    yl2 = LBOUND(B,2) 
    yu2 = UBOUND(B,2) 

    !$omp target
    !$omp teams num_teams(xu1-xl1+1)
    !$omp distribute
    DO i=xl1, xu1
        !$omp parallel do
        DO j=yl2, yu2
            C(i, j) = 0
            DO k=yl1, yu1
                C(i, j) = C(i, j) + A(i, k) * B(k, j)
            END DO
        END DO
    END DO
    !$omp end teams
    !$omp end target

    !$omp target update from(C)
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

    data = {"copyinout": [], "copyin":[], "copyout":[], "alloc":[]}

    if testname == "vecadd1d":
        data["copyin"].append(np.reshape(np.arange(N1, dtype=np.int64), (N1,), order=order))
        data["copyin"].append(np.reshape(np.arange(N1, dtype=np.int64) * 2, (N1,), order=order))
        data["copyout"].append(np.reshape(np.zeros(N1, dtype=np.int64), (N1,), order=order))
    elif testname == "matmul":
        data["copyin"].append(np.reshape(np.arange(100, dtype=np.float64), (4, 25), order=order))
        data["copyin"].append(np.reshape(np.arange(100, dtype=np.float64) * 2, (25, 4), order=order))
        data["copyout"].append(np.reshape(np.zeros(16, dtype=np.float64), (4, 4), order=order))

    elif testname == "vecadd3d":
        data["copyin"].append(np.reshape(np.arange(100, dtype=np.int64), N3, order=order))
        data["copyin"].append(np.reshape(np.arange(100, dtype=np.int64) * 2, N3, order=order))
        data["copyout"].append(np.reshape(np.zeros(100, dtype=np.int64), N3, order=order))

    else:
        assert False

    return data


def check_result(testname, data):

    if testname == "vecadd1d":
        assert np.array_equal(data["copyout"][0], data["copyin"][0] + data["copyin"][1])

    elif testname == "matmul":
        assert np.array_equal(data["copyout"][0], np.matmul(data["copyin"][0], data["copyin"][1]))

    elif testname == "vecadd3d":
        assert np.array_equal(data["copyout"][0], data["copyin"][0] + data["copyin"][1])

    else:
        assert False


###################
#  Functions
###################

def get_testdata(testname, lang):
    #return _gendata(testname, lang), specs[testname]

#def _gendata(testname, lang):

    N1 = 100
    N3 = (2, 5, 10)

    if lang == "cpp":
        order = "C"

    elif lang == "fortran":
        order = "F"

    data = {"copyinout": [], "copyin":[], "copyout":[], "alloc":[]}
    conf = []

    if testname == "vecadd1d":
        data["copyin"].append(np.reshape(np.arange(N1, dtype=np.int64), (N1,), order=order))
        data["copyin"].append(np.reshape(np.arange(N1, dtype=np.int64) * 2, (N1,), order=order))
        data["copyout"].append(np.reshape(np.zeros(N1, dtype=np.int64), (N1,), order=order))
        conf.append(1)
        conf.append(N1)
    elif testname == "matmul":
        data["copyin"].append(np.reshape(np.arange(100, dtype=np.float64), (4, 25), order=order))
        data["copyin"].append(np.reshape(np.arange(100, dtype=np.float64) * 2, (25, 4), order=order))
        data["copyout"].append(np.reshape(np.zeros(16, dtype=np.float64), (4, 4), order=order))
        conf.append(1)
        conf.append((25, 4))
    elif testname == "vecadd3d":
        data["copyin"].append(np.reshape(np.arange(100, dtype=np.int64), N3, order=order))
        data["copyin"].append(np.reshape(np.arange(100, dtype=np.int64) * 2, N3, order=order))
        data["copyout"].append(np.reshape(np.zeros(100, dtype=np.int64), N3, order=order))
        conf.append(1)
        conf.append(N3)
    else:
        assert False

    return data, specs[testname], conf


def assert_testdata(testname, data):
    check_result(testname, data)
