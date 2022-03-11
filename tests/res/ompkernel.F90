MODULE modompkernel

public runkernel

CONTAINS

INTEGER (C_INT64_T) FUNCTION runkernel(X, Y, Z) BIND(C, name="runkernel")
    USE, INTRINSIC :: ISO_C_BINDING

    REAL(C_DOUBLE), DIMENSION(2, 3), INTENT(IN) :: X, Y
    REAL(C_DOUBLE), DIMENSION(2, 3), INTENT(OUT) :: Z

    INTEGER i, j

    !$omp target teams num_teams(2)
    !$omp distribute
    DO i=1, 2
        !$omp parallel do
        DO j=1, 3
            Z(i, j) = X(i, j) + Y(i, j)
        END DO
        !$omp end parallel do
    END DO
    !$omp end target teams

    runkernel = 0

END FUNCTION

END MODULE
