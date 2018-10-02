
# cython: infer_types=True

import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)

cpdef range_corrected( np.ndarray[DTYPE_t, ndim=2] matrix, np.ndarray [DTYPE_t, ndim=1] rang, int step ):
    cdef int i
    cdef int xdim = matrix.shape[0]
    cdef int ydim = matrix.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros([xdim,ydim], dtype=DTYPE)

    for i in range(0,ydim,step):
        result [:,i:i+step] = matrix[:,i:i+step] * rang[i]
    return result

# def range_corrected(double [:,:] matrix, dobule [:] range ):
