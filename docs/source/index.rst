.. AccelPy documentation master file, created by
   sphinx-quickstart on Fri Apr 22 13:04:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|project|
===================================

|desc|

**NOTE** : |project| is under development, not for production-use.


Introduction
--------------

AccelPy is a Python module that allows developers conviniently run accelerator codes written in Fortran and C/C++. As of this writing, AccelPy supports programming models of "Openmp target", "OpenAcc", "CUDA", "HIP", "Openmp", and plain "Fortran" as well as "C/C++".

Instead of having its own compilation capability, AccelPy relys on compilers installed on the system where AccelPy runs. AccelPy supports several well-known compilers such as GNU, IBM XL, INTEL, PGI, CRAY, AMD, NVIDIA.

User may create an application in Python first and may convert time-consuming parts of the application one by one to accelerated compiler-based codes in the programming models mentioned above. AccelPy can be considered as a glue that makes the acclereated code to something usable by Python codes through utilizing compilers.

To make this works as simplely as possible, AccelPy only accepts data used in the accelerated codes in the form of NumPy array or something that can be conveted to Numpy array. This helps AccelPy to collect essential information about the data such as data type, shape of the array, and so on.

While AccelPy handles many of boiler-plating tasks to make some Fortran/C/C++ codes work with Python code, user still have to provide core part of algorithm in one or more programming models mentioned above. This user-provided code is called "kernel" in AccelPy term and user can create a kernel that supports multiple programming models that can be selected at runtime.


Capabilities
--------------

* Converts algorithm written in "Fortran/C/C++" into a shared library callable from Python
* Supports programming models of "Openmp target", "OpenAcc", "CUDA", "HIP", "Openmp", and plain "Fortran" as well as "C/C++"
* Handles data movement between CPU and GPU so that user can write GPU code only.
* Selects a programming model of a kernel at runtime


Example
--------------

In this example, we will add two 2-dimensional vectors(x, y) element-wisely and save their sum in a vector(z). User can define acceleration task using Accel class. The named arguments of "copyin" and "copyout"ensure that the specified data are available to target device such as CPU or GPU. In this example, two Numpy ndarray "x" and "y" are available in the target device as input variables and "z" as an output varilable. In the case of CPU, those optional argument do nothing because the data are already available on CPU memory space. Howver, in the case of GPU, AccelPy does allocate and copy the data to GPU.

"launch" function compiles and run the kernel that is specified in the file of "vecadd.knl". The details of the kernel file is explained below. The launch function also needs the input data arguments of "x", "y", and "z". However, because these data are already available in target device through the "copyin", there is no actual data movement during the execution of "launch" function.

Finally, "stop" function of Accel finalize and copy back the data specified by "copyout" so that the Numpy ndarray "z" has the result from the kernel execution specified by "vecadd.knl". 


**main.py**
::

    import numpy as np
    from accelpy import Accel, Kernel

    vecadd = "vecadd.knl"

    N1 = 10
    N2 = 20

    x = np.ones((N1, N2), order="F")
    y = np.ones((N1, N2), order="F")
    z = np.zeros((N1, N2), order="F")


    acctarget = "openacc" # "omptarget"

    accel = Accel(accel=acctarget, copyin=(x, y), copyout=(z,))

    accel.launch(Kernel(vecadd), x, y, z)

    accel.stop()

    print("SUCCESS" if np.array_equal(z, x+y) else "FAIL")


"vecadd.knl" is an AccelPy Kernel Specification that defines the kernel. To make is simple, it shows Fortran versions only. However, AccelPy support C++ kernel implementation also.

The pair of square brackets is a kernel section. In the file, there are two kernel section. One defines kernel for Fortran and Openmp target and the other defines for Openacc. User can specify a particular target of kernel type. Otherwise AccelPy select one of available kernel types. The names of the programming models indicates that this section implements plain fortran, openacc fortran, and openmp target fortran implementation. AccelPy also supports plain c++, openacc c++, openmp c++, openmp target c++, cuda, hip, and openmp fortran. Next three arguments are the names of input arguments to this kernel.

The next lines implements the kernel of adding two vectors element by element. Because AccelPy handles the data movements for the inputs and the outputs of this kernel, user only need to focus on the implementation of algorithm in the kernel body.

**vecadd.knl**
::

	[fortran, omptarget_fortran: a, b, c]

	INTEGER i, j

	!$omp target teams num_teams(SIZE(a, 1))
	!$omp distribute
	DO i=LBOUND(a,1), UBOUND(a,1)
		!$omp parallel do
		DO j=LBOUND(a,2), UBOUND(a,2)
			c(i, j) = a(i, j) + b(i, j)
		END DO
	END DO
	!$omp end target teams

	[openacc_fortran: a, b, c]

	INTEGER i, j

	!$acc parallel num_gangs(SIZE(a, 1)), vector_length(SIZE(a, 2))
	!$acc loop gang
	DO i=LBOUND(a,1), UBOUND(a,1)
		!$acc loop vector
		DO j=LBOUND(a,2), UBOUND(a,2)
			c(i, j) = a(i, j) + b(i, j)
		END DO
	END DO
	!$acc end parallel


Installation
----------------

The easiest way to install accelpy is to use the pip python package manager.

        >>> pip install accelpy

You can install accelpy from github code repository if you want to try the latest version.

        >>> git clone https://github.com/grnydawn/accelpy.git
        >>> cd accelpy
        >>> python setup.py install


.. only:: html

    :Release: |release|
    :Date: |today|

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
