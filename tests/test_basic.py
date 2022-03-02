
from accelpy import Kernel, AccelData
from pytest import mark, fixture
from compilers import testable, not_tested
from testdata import get_testdata, assert_testdata

debug = False

@fixture(autouse=True)
def run_around_tests():
    #files_before = # ... do something to check the existing files
    # A test function will be run at this point
    yield
    # Code that will run after your test, for example:
    #files_after = # ... do something to check the existing files
    #assert files_before == files_after

@mark.parametrize("acctype, compile", testable)
def ttest_first(acctype, compile):

    lang = "fortran" if "fortran" in acctype else "cpp"

    data, spec  = get_testdata("vecadd1d", lang)

    kernel = Kernel(spec, acctype=acctype, compile=compile, debug=debug)

    task = kernel.launch(*data)

    task.wait()

    kernel.stop()

    assert_testdata("vecadd1d", data)

@mark.parametrize("acctype, compile", testable)
def ttest_vecadd3d(acctype, compile):

    lang = "fortran" if "fortran" in acctype else "cpp"

    data, spec  = get_testdata("vecadd3d", lang)

    kernel = Kernel(spec, acctype=acctype, compile=compile, debug=debug)

    task = kernel.launch(*data)

    task.wait()

    kernel.stop()

    assert_testdata("vecadd3d", data)


@mark.parametrize("acctype, compile", testable)
def ttest_matmul(acctype, compile):

    lang = "fortran" if "fortran" in acctype else "cpp"

    data, spec  = get_testdata("matmul", lang)

    kernel = Kernel(spec, acctype=acctype, compile=compile, debug=debug)

    task = kernel.launch(*data)

    task.wait()

    kernel.stop()

    assert_testdata("matmul", data)


@mark.parametrize("acctype, compile", testable)
def test_acceldata(acctype, compile):


    lang = "fortran" if "fortran" in acctype else "cpp"

    data, spec  = get_testdata("matmul", lang)

    kernel = Kernel(spec, acctype=acctype, compile=compile, debug=debug)
    #kernel = Kernel(spec, acctype=acctype, compile="-Minfo -ta=tesla", debug=debug)

    #if lang == "cpp":
    #    import pdb; pdb.set_trace()

    accel = AccelData(kernel, mapto=data[:2], mapfrom=data[2:],
                debug=debug, acctype=acctype, compile=compile)

    kernel.launch(*data)

    accel.stop()

    assert_testdata("matmul", data)


#def test_allcompilers():
#    assert not_tested == []
