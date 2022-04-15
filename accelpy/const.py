"Constant module"

name            = "accelpy"
version         = "0.6.11"
description     = "Scalable Accelerator Interface in Python"
long_description= "Scalable Accelerator Interface in Python"
author          = "Youngsung Kim"
vendors         = ("cray", "amd", "nvidia", "intel", "pgi", "ibm", "gnu")

NOCACHE         = 0
FILECACHE       = 1
MEMCACHE        = 2 

NODEBUG         = 0
MINDEBUG        = 1
MAXDEBUG        = 2

NOPROF          = 0
MINPROF         = 1
MAXPROF         = 2

# priorities
lang_priority = ("fortran", "cpp")
accel_priority = ("omptarget_fortran", "omptarget_cpp", "openmp_fortran",
                    "openmp_cpp", "openacc_fortran", "openacc_cpp", "hip",
                    "cuda", "fortran", "cpp")
vendor_priority = ("cray", "amd", "nvidia", "intel", "pgi", "ibm", "gnu")

