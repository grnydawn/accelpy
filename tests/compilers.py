"detect locally available compilers"

from accelpy.compiler import Compiler

testable = []
not_tested = []

for lang, accels in Compiler.avails.items():
    for accel, vendors in accels.items():
        for vendor, compcls in vendors.items():
            try:
                compobj = compcls()
                testable.append((accel, vendor))

            except Exception as err:
                not_tested.append((accel, vendor))
