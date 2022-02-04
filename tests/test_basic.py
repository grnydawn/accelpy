
from accelpy import Kernel, Spec
from pytest import mark
from compilers import testable, not_tested
from testdata import get_testdata, assert_testdata

@mark.parametrize("accel, compile", testable)
def test_first(accel, compile):

    data, spec  = get_testdata("vecadd1d")

    kernel = Kernel(Spec(spec), accel=accel, compile=compile, debug=True)

    task = kernel.launch(*data)

    task.wait()

    kernel.stop()

    assert_testdata("vecadd1d", data)


#def test_allcompilers():
#    assert not_tested == []
