# cython_extension/error_function.pyx

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np

from .dl85Optimizer cimport RCover


cdef class ArrayIterator:
    # DO NOT redeclare attributes here - they're already in the .pxd file
    # cdef RCover* arr        <-- REMOVE
    # cdef RCover.iterator it <-- REMOVE
    # cdef bool trans_loop    <-- REMOVE

    def __init__(self, trans):
        self.trans_loop = trans

    def __iter__(self):
        return self

    def __next__(self):
        if self.trans_loop:
            if self.it.wordOrder < self.arr.limit.top():
                val = deref(self.it)
                self.it = inc(self.it)
                return val
            else:
                raise StopIteration()
        else:
            val = deref(self.it)
            self.it = inc(self.it)
            return val

    cdef void init_iterator(self):
        self.it = self.arr.begin(self.trans_loop)


cdef public object wrap_array(RCover *ar, bool trans):
    tid_python_object = ArrayIterator(trans)
    tid_python_object.arr = ar
    tid_python_object.init_iterator()
    return tid_python_object


cdef public float* call_python_tid_error_class_function(py_function, RCover *ar):
    error_class_array = np.array(py_function(wrap_array(ar, True)), dtype=np.float32)
    if not error_class_array.flags['C_CONTIGUOUS']:
        error_class_array = np.ascontiguousarray(error_class_array)
    cdef float[::1] error_class_view = error_class_array
    return &error_class_view[0]


cdef public float* call_python_support_error_class_function(py_function, RCover *ar):
    error_class_array = np.array(py_function(wrap_array(ar, False)), dtype=np.float32)
    if not error_class_array.flags['C_CONTIGUOUS']:
        error_class_array = np.ascontiguousarray(error_class_array)
    cdef float[::1] error_class_view = error_class_array
    return &error_class_view[0]


cdef public float call_python_tid_error_function(py_function, RCover *ar):
    return py_function(wrap_array(ar, True))