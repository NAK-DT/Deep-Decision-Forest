# cython_extension/error_function.pxd

from libcpp cimport bool
from libcpp.stack cimport stack
from .dl85Optimizer cimport RCover

# Declare external C++ DataManager class
cdef extern from "../core/src/dataManager.h":
    cdef cppclass DataManager:
        int getNClasses()
        int* getSupports()

# Declare Python-wrapped Cython class interface (no bodies)
cdef class ArrayIterator:
    cdef RCover* arr
    cdef RCover.iterator it
    cdef bool trans_loop
    # In .pxd files, just declare method signatures without 'def'
    cdef void init_iterator(self)
    # Python-accessible methods should be declared differently in .pxd files
    # The actual implementations go in the .pyx file

# Public function declarations
cdef public object wrap_array(RCover *ar, bool trans)
cdef public float* call_python_tid_error_class_function(object py_function, RCover* ar)
cdef public float* call_python_support_error_class_function(object py_function, RCover* ar)
cdef public float call_python_tid_error_function(object py_function, RCover* ar)