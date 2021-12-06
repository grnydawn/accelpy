.. accelpy documentation master file, created by
   sphinx-quickstart on Sat Dec  4 16:54:01 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. only:: html

    :Release: |release|
    :Date: |today|

Welcome to accelpy's documentation!
===================================

**accelpy** is a light-weight Python interface package to native programming models. It makes use of multiple programming models including Cuda, Hip, OpenAcc, OpenMp, C++, and Fortran. **accelpy** provides Python programmers with a stable and Pythonic API for using accelerators, while maximizs its capability and flexibility through offloading the computationally intensitve part to a native code, or kernel.

**accelpy** is not the only one that tackles the accelerator programming in Python. Numba transforms Cuda-like Python code into GPU binary. If you are interested in machine learning, you may already use GPUs through PyTorch or TensorFlow. Nvidia provides users with a series of Cuda-python bindings.

**accelpy** has unique features compared to the other similar technologies.

It utilizes currently available accelerator and traditional programming models. Each programming models have matured for decades and gained their own advantages in particular problem domains. Python is awesome programming language but it may be hard to pull all of power of newer hardwares while writing everything in Python, in my opinion. For example, it would be the safe bet that the newest Nvidia GPU feature may be available in Cuda first.

Related to above characteristic of **accelpy**, it is relatively easy to support newer programming models or newer hardware because **accelpy** does not need to create a wheel by itself. It may be a month-long or even a week-long task to support a new programming model at least working version. In other words, **accelpy** is almost future-proof in terms of programmig model.

Using **accelpy** has two sides: Python side and native programming side. In general, each side has differnt programming goals. On Python side, we want usability among others, while performance may be prefered at native programming side. By not mixing those two sides in **accelpy**, user may be better focus on its own goals at each sides. Python programmer using **accelpy** does not need to know the details of the native programming because **accelpy** abstractized the it through simple models of input & output data, accelerator, and order( native programming representation )



Maintaining multple sources in one application to support multiple hardware is painful and expensive. Adding preprocessing blocks for each hardware in source files makes it hard to read and Traditional approache to achieve this is to he
- practical solution to one-source multiuse: gradual migration, dynamic backup mechanism
- algorithm hot-swap (not supported yet)

To use **accelpy**, conceptually, user defines what an accelerator does by providing **accelpy** with an "order", computational code in multiple native programming models and inputs & outputs. And the user executes the "order" to get results.

Practically, **accelpy** generates and compiles a source code based on the "order" and inputs & outputs to build a shared library. Once the shared library is built, **accelpy** sends the input data to accelerator, runs the "order" in the generated shared library, and finally receives the result from executing the "order" to the output variable(s). In other words, **accelpy** takes the responsibility of native code interface, data movement between host and accelerator, and accelerator execution control.

**accelpy is not for production use yet.**

An example of adding two vectors in Cuda, Hip, OpenAcc, or OpenMp:

::

        import numpy as np
        from accelpy import Accel, Order

        N = 100
        a = np.arange(N)                # input a
        b = np.arange(N)                # input b
        c = np.zeros(N, dtype=np.int64) # output c

        # define acceleration task in one or more programming models in either a string or a file
        vecadd = """
        set_argnames(("a", "b"), "c")

        [hip, cuda]
            int id = blockIdx.x * blockDim.x + threadIdx.x;
            if(id < a.size) c(id) = a(id) + b(id);

        [openacc_cpp]
            #pragma acc loop gang worker vector
            for (int id = 0; id < a.shape[0]; id++) {
                c(id) = a(id) + b(id);
            }

        [openmp_fortran]
            INTEGER id

            !$omp do
            DO id=1, a_attr%shape(1)
                c(id) = a(id) + b(id)
            END DO
            !$omp end do
        """

        # create a task to be offloaded to an accelerator
        # with an order, inputs(a, b), and an output(c)
        accel = Accel(a, b, Order(vecadd), c)

        # asynchronously launch N-parallel work 
        accel.run(N)

        # do Python work here while accelerator is working

        # implicitly copy the calculation result to the output array "c"
        accel.stop()

        assert np.array_equal(c, a + b)

Assuming that at least one compiler of the programming models (and a hardware) is available, the "vecadd order" will be compiled and executed on either a GPU or a CPU.

The easiest way to install **accelpy** is to use the pip python package manager.

        >>> pip install accelpy

Source code: `https://github.com/grnydawn/accelpy <https://github.com/grnydawn/accelpy/>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
