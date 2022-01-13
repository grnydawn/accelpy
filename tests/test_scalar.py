"""accelpy basic tests"""

import numpy as np
import pytest
from accelpy import Accel, Order


#######################
# Order definitions
#######################
test_accels = (
#    ("cpp", "gnu"),
#    ("cpp", "cray"),
#    ("cpp", "amd"),
#    ("cpp", "ibm"),
#    ("cpp", "pgi"),
#    ("cpp", "intel"),
#    ("fortran","gnu"),
#    ("fortran", "cray"),
#    ("fortran", "amd"),
#    ("fortran", "ibm"),
#    ("fortran", "pgi"),
#    ("fortran", "intel"),
#    ("openmp_cpp", "gnu"), # GNU compiler error on Summit
#    ("openmp_cpp", "pgi"),
#    ("openmp_cpp", "ibm"),
#    ("openmp_cpp", "cray"),
#    ("openmp_cpp", "amd"),
#    ("openmp_fortran", "gnu"),
#    ("openmp_fortran", "pgi"),
#    ("openmp_fortran", "ibm"),
#    ("openmp_fortran", "cray"), # OpenMP parallel attempted from non-OpenMP thread On Spock
#    ("openmp_fortran", "amd"),
#    ("openacc_cpp", "pgi"),
#    ("openacc_cpp", "gnu"), # GNU compiler error on Summit
#    ("openacc_cpp", "cray"),
#    ("openacc_fortran", "pgi"),
#    ("openacc_fortran", "gnu"), # GNU compiler error on Summit
#    ("openacc_fortran", "cray"),
#    ("hip", "amd"),
    ("cuda", "nvidia"),

)

order_sum = """

set_argnames(("a", "b", "c"), ("out",))

[cpp]
    out(0) = a + b + c;

[fortran]
    out(1) = a + b + c

[openmp_cpp]
    out(0) = a + b + c;

[openmp_fortran]
    out(1) = a + b + c

[openacc_cpp]
    out(0) = a + b + c;

[openacc_fortran]
    out(1) = a + b + c

[cuda]
    out(0) = a + b + c;

"""

@pytest.mark.parametrize("accel, comp", test_accels)
def test_double(accel, comp):

    a, b, c = 1.0, 2.0, 3.0
    out = [0.]

    accel = Accel(a, b, c, Order(order_sum), out,
                    kind=[accel], compile=[comp])

    accel.run(1)

    accel.stop()

    assert out[0] == a + b + c

