"""accelpy core module
"""

import abc
#
#name            = "accelpy"
#version         = "0.3.15"
#description     = "Scalable Accelerator Interface in Python"
#long_description = "Scalable Accelerator Interface in Python"
#author          = "Youngsung Kim"

class Object(abc.ABC):

    def __new__(cls, *vargs, **kwargs):

        obj = super(Object, cls).__new__(cls)

        return obj
