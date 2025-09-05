from libcpp.string cimport string
from libcpp cimport bool, nullptr
from libcpp.vector cimport vector
from libcpp.functional cimport function
from libcpp.stack cimport stack
from cython.operator cimport dereference as deref, preincrement as inc
import numpy as np

cdef class ArrayIterator:
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

cdef extern from "../core/src/cache.h":
    cpdef enum CacheType:
        CacheTrieItemset,
        CacheHashItemset,
        CacheHashCover

    cpdef enum WipeType:
        All,
        Subnodes,
        Reuses

cdef extern from "py_tid_error_class_function_wrapper.h":
    cdef cppclass PyTidErrorClassWrapper:
        PyTidErrorClassWrapper()
        PyTidErrorClassWrapper(object)

cdef extern from "py_support_error_class_function_wrapper.h":
    cdef cppclass PySupportErrorClassWrapper:
        PySupportErrorClassWrapper()
        PySupportErrorClassWrapper(object)

cdef extern from "py_tid_error_function_wrapper.h":
    cdef cppclass PyTidErrorWrapper:
        PyTidErrorWrapper()
        PyTidErrorWrapper(object)

cdef extern from "../core/src/dl85.h":
    string launch ( float* supports,
                    int ntransactions,
                    int nattributes,
                    int nclasses,
                    int *data,
                    int *target,
                    int maxdepth,
                    int minsup,
                    float maxError,
                    bool stopAfterError,
                    PyTidErrorClassWrapper tids_error_class_callback,
                    PySupportErrorClassWrapper supports_error_class_callback,
                    PyTidErrorWrapper tids_error_callback,
                    float* in_weights,
                    bool tids_error_class_is_null,
                    bool supports_error_class_is_null,
                    bool tids_error_is_null,
                    bool infoGain,
                    bool infoAsc,
                    bool repeatSort,
                    int timeLimit,
                    bool verbose_param,
                    CacheType cache_type,
                    int cache_size,
                    WipeType wipe_type,
                    float wipe_factor,
                    bool with_cache,
                    bool useSpecial,
                    bool use_ub,
                    bool similarlb,
                    bool dynamic_branching,
                    bool similar_for_branching,
                    bool from_cpp) except +

def solve(data, target, tec_func_=None, sec_func_=None, te_func_=None, max_depth=1, min_sup=1, example_weights=[], max_error=0, stop_after_better=False, time_limit=0, verb=False, desc=False, asc=False, repeat_sort=False, predictor=False, cachetype=CacheTrieItemset, cachesize=0, wipetype=Reuses, wipefactor=0.5, withcache=True, usespecial=True, useub=True, similar_lb=False, dyn_branch=False, similar_for_branching=True):
    cdef PyTidErrorClassWrapper tec_func = PyTidErrorClassWrapper(tec_func_)
    tec_null_flag = True
    if tec_func_ is not None:
        tec_null_flag = False
    cdef PySupportErrorClassWrapper sec_func = PySupportErrorClassWrapper(sec_func_)
    sec_null_flag = True
    if sec_func_ is not None:
        sec_null_flag = False
    cdef PyTidErrorWrapper te_func = PyTidErrorWrapper(te_func_)
    te_null_flag = True
    if te_func_ is not None:
        te_null_flag = False
    data = data.astype('int32')
    ntransactions, nattributes = data.shape
    data = data.transpose()
    classes, supports = np.unique(target, return_counts=True)
    nclasses = len(classes)
    supports = supports.astype('float32')
    if np.array_equal(data, data.astype('bool')) is False:
        raise ValueError("Bad input type. DL8.5 actually only supports binary (0/1) inputs")
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    cdef int [:, ::1] data_view = data
    cdef int *data_matrix = &data_view[0][0]
    cdef int [::1] target_view
    cdef int *target_array = NULL
    if target is not None:
        target = target.astype('int32')
        if not target.flags['C_CONTIGUOUS']:
            target = np.ascontiguousarray(target)
        target_view = target
        target_array = &target_view[0]
    else:
        nclasses = 0
    if not supports.flags['C_CONTIGUOUS']:
        supports = np.ascontiguousarray(supports)
    cdef float [::1] supports_view = supports
    cdef float [::1] ex_weights_view
    cdef float *ex_weights_pointer = NULL
    if len(example_weights) > 0:
        ex_weights = np.asarray(example_weights, dtype=np.float32)
        if not ex_weights.flags['C_CONTIGUOUS']:
            ex_weights = np.ascontiguousarray(ex_weights)
        ex_weights_view = ex_weights
        ex_weights_pointer = &ex_weights_view[0]
    if max_error < 0:
        stop_after_better = False
    info_gain = not (desc == False and asc == False)
    out = launch(supports = &supports_view[0], ntransactions = ntransactions, nattributes = nattributes, nclasses = nclasses, data = data_matrix, target = target_array, maxdepth = max_depth, minsup = min_sup, maxError = max_error, stopAfterError = stop_after_better, tids_error_class_callback = tec_func, supports_error_class_callback = sec_func, tids_error_callback = te_func, in_weights = ex_weights_pointer, tids_error_class_is_null = tec_null_flag, supports_error_class_is_null = sec_null_flag, tids_error_is_null = te_null_flag, infoGain = info_gain, infoAsc = asc, repeatSort = repeat_sort, timeLimit = time_limit, verbose_param = verb, cache_type = cachetype, cache_size = cachesize, wipe_type = wipetype, wipe_factor = wipefactor, with_cache = withcache, useSpecial = usespecial, use_ub = useub, similarlb = similar_lb, dynamic_branching = dyn_branch, similar_for_branching = similar_for_branching, from_cpp = False)
    return out
