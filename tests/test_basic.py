
from accelpy import Kernel, Spec
from pytest import mark
from compilers import testable, not_tested
from testdata import get_testdata, assert_testdata

@mark.parametrize("accel, compile", testable)
def ttest_first(accel, compile):

    lang = "fortran" if "fortran" in accel else "cpp"

    data, spec  = get_testdata("vecadd1d", lang)

    kernel = Kernel(Spec(spec), accel=accel, compile=compile, debug=True)

    task = kernel.launch(*data)

    task.wait()

    kernel.stop()

    assert_testdata("vecadd1d", data)

@mark.parametrize("accel, compile", testable)
def ttest_vecadd3d(accel, compile):

    lang = "fortran" if "fortran" in accel else "cpp"

    data, spec  = get_testdata("vecadd3d", lang)

    kernel = Kernel(Spec(spec), accel=accel, compile=compile, debug=True)

    task = kernel.launch(*data)

    task.wait()

    kernel.stop()

    assert_testdata("vecadd3d", data)


@mark.parametrize("accel, compile", testable)
def test_matmul(accel, compile):

    lang = "fortran" if "fortran" in accel else "cpp"

    data, spec  = get_testdata("matmul", lang)

    kernel = Kernel(Spec(spec), accel=accel, compile=compile, debug=True)

    task = kernel.launch(*data)

    task.wait()

    kernel.stop()

    assert_testdata("matmul", data)


#def test_allcompilers():
#    assert not_tested == []
