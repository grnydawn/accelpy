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
    ("fortran", "ibm"),
#    ("fortran", "pgi"),
#    ("fortran", "intel"),
)

order_sum = """

set_argnames(("a", "b", "c"), ("out",))

[cpp]
    out(0) = a + b + c;

[fortran]
    out(1) = a + b + c;

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

