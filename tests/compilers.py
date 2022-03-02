"detect locally available compilers"

from accelpy.compiler import Compiler

excepts = {
    "cray": ["openacc_fortran", "openmp_fortran", "omptarget_cpp"],
}

testable = []
not_tested = []

for lang, accels in Compiler.avails.items():
    for accel, vendors in accels.items():
        for vendor, compcls in vendors.items():
            if vendor in excepts and accel in excepts[vendor]:
                continue

            try:
                compobj = compcls()
                testable.append((accel, vendor))

            except Exception as err:
                not_tested.append((accel, vendor))


#testable = [("cuda", "nvidia")]
#testable = [("hip", "amd")]
#testable = [("cpp", "cray")]
#testable = [("omptarget_fortran", "cray")]
testable = [("openacc_fortran", "cray")]
#testable = [("openacc_fortran", "gnu")]
#testable = [("omptarget_fortran", "gnu")]
#testable = [("omptarget_fortran", "amd")]
#testable = [("openacc_fortran", "amd")]
#testable = [("openacc_fortran", "pgi")]
#testable = [("openmp_cpp", "cray")]
#testable = [("omptarget_cpp", "amd")]
#testable = [("omptarget_cpp", "cray")]
#testable = [("omptarget_cpp", "gnu")]
#testable = [("openacc_cpp", "pgi")]
#testable = [("omptarget_cpp", "pgi")]
#testable = [("openacc_fortran", "pgi")]
#testable = [("omptarget_fortran", "pgi")]
