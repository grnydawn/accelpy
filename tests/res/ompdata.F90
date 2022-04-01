module modompdata
USE, INTRINSIC :: ISO_C_BINDING

public dataenter, dataexit

contains

INTEGER (C_INT64_T) FUNCTION dataenter(X, Y, Z) BIND(C, name="dataenter")
    USE, INTRINSIC :: ISO_C_BINDING

    REAL(C_DOUBLE), DIMENSION(2, 3), INTENT(IN) :: X, Y, Z

    !!$omp target enter data map(to: X(:,:), Y(:,:)) map(alloc: Z(:,:))
    !!$omp target enter data map(to: X, Y) map(alloc: Z)

    dataenter = 0

END FUNCTION

INTEGER (C_INT64_T) FUNCTION dataexit(Z) BIND(C, name="dataexit")
    USE, INTRINSIC :: ISO_C_BINDING

    REAL(C_DOUBLE), DIMENSION(2, 3), INTENT(OUT) :: Z

    !!$omp target exit data map(from: Z(:,:))
    !!$omp target exit data map(from: Z)

    dataexit = 0

END FUNCTION

end module
