from libcpp cimport bool
from libcpp.stack cimport stack

cdef extern from "../core/src/rCover.h":
    cdef cppclass RCover:
        cppclass iterator:
            int wordOrder
            int value
            iterator()
            int operator*()
            iterator operator++()
        stack[int] limit
        iterator begin(bool)
        iterator end(bool)

cdef extern from "../core/src/dataManager.h":
    cdef cppclass DataManager:
        int getNClasses()
        int* getSupports()

cdef class ArrayIterator:
    cdef RCover* arr
    cdef RCover.iterator it
    cdef bool trans_loop
    cdef void init_iterator(self)

cdef public object wrap_array(RCover *ar, bool trans)
cdef public float* call_python_tid_error_class_function(object py_function, RCover* ar)
cdef public float* call_python_support_error_class_function(object py_function, RCover* ar)
cdef public float call_python_tid_error_function(object py_function, RCover* ar)
