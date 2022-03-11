MODULE modacckernel

public runkernel

CONTAINS

INTEGER (C_INT64_T) FUNCTION runkernel(X, Y, Z) BIND(C, name="runkernel")
    USE, INTRINSIC :: ISO_C_BINDING

    REAL(C_DOUBLE), DIMENSION(2, 3), INTENT(IN) :: X, Y
    REAL(C_DOUBLE), DIMENSION(2, 3), INTENT(OUT) :: Z

    INTEGER i, j

    !$acc parallel num_gangs(2) num_workers(3) 
    !$acc loop gang
    DO i=1, 2
        !$acc loop worker
        DO j=1, 3
            Z(i, j) = X(i, j) + Y(i, j)
        END DO
    END DO
    !$acc end parallel

    runkernel = 0

END FUNCTION

END MODULE
