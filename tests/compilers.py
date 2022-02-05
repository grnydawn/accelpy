"detect locally available compilers"

from accelpy.compiler import Compiler

excepts = {
    "cray": ["fortran", "openacc_fortran"],
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


#testable = [("openacc_fortran", "gnu")]
