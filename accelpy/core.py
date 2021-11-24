"""accelpy core module
"""

import abc


name            = "accelpy"
version         = "0.2.1"
description     = "Scalable Accelerator Interface in Python"
long_description = """
T.B.D.
"""


class Object(abc.ABC):

    def __new__(cls, *vargs, **kwargs):

        obj = super(Object, cls).__new__(cls)

        return obj
