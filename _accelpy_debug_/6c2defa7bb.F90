


INTEGER (C_INT64_T) FUNCTION accelpy_varmap_a (data) BIND(C, name="accelpy_varmap_a")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : a
    IMPLICIT NONE

    INTEGER (C_INT64_T), DIMENSION(100), INTENT(IN), TARGET :: data

    a => data

    accelpy_varmap_a = 0

END FUNCTION


INTEGER (C_INT64_T) FUNCTION accelpy_varmap_b (data) BIND(C, name="accelpy_varmap_b")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : b
    IMPLICIT NONE

    INTEGER (C_INT64_T), DIMENSION(100), INTENT(IN), TARGET :: data

    b => data

    accelpy_varmap_b = 0

END FUNCTION


INTEGER (C_INT64_T) FUNCTION accelpy_varmap_c (data) BIND(C, name="accelpy_varmap_c")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : c
    IMPLICIT NONE

    INTEGER (C_INT64_T), DIMENSION(100), INTENT(IN), TARGET :: data

    c => data

    accelpy_varmap_c = 0

END FUNCTION


INTEGER (C_INT64_T) FUNCTION accelpy_start() BIND(C, name="accelpy_start")

    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : a, b, c
    IMPLICIT NONE

    accelpy_start = accelpy_kernel(a, b, c)

CONTAINS


INTEGER (C_INT64_T) FUNCTION accelpy_kernel(a, b, c)
    USE, INTRINSIC :: ISO_C_BINDING
    IMPLICIT NONE

    INTEGER (C_INT64_T), DIMENSION(:), INTENT(INOUT) :: a
INTEGER (C_INT64_T), DIMENSION(:), INTENT(INOUT) :: b
INTEGER (C_INT64_T), DIMENSION(:), INTENT(INOUT) :: c

        INTEGER id

    print *, a, b, c
    !DO id=LBOUND(a,1), UBOUND(a,1)
    !    c(id) = a(id) + b(id)
    !END DO


    accelpy_kernel = 0

END FUNCTION


END FUNCTION

INTEGER (C_INT64_T) FUNCTION accelpy_stop() BIND(C, name="accelpy_stop")
    USE, INTRINSIC :: ISO_C_BINDING
    USE accelpy_global, ONLY : a, b, c
    IMPLICIT NONE

    accelpy_stop = 0

END FUNCTION
