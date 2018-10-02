


import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def range_corrected( np.ndarray [DTYPE_t, ndim=2] matrix,
                    np.ndarray [DTYPE_t, ndim=1] rang):
    cdef int i,j
    cdef int rdim = matrix.shape[0]
    cdef int cdim = matrix.shape[1]
    #cdef np.ndarray[DTYPE_t, ndim=2] result = np.empty_like(matrix, dtype=DTYPE)

    for i in range(rdim):
        for j in range(cdim):
            matrix [i,j] *= rang[j]
    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
def mVolts ( np.ndarray [np.int64_t, ndim=2] matrix,
            np.ndarray[DTYPE_t, ndim=1] inputRange,
            np.ndarray[DTYPE_t, ndim=1] ADCBits,
            np.ndarray[DTYPE_t, ndim=1] shotNumber):

    cdef int i,j
    cdef DTYPE_t power
    cdef int const1 = 1000
    cdef int const2 = 2
    cdef int rdim = matrix.shape[0]
    cdef int cdim = matrix.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.empty_like(matrix, dtype=DTYPE)

    for i in range(rdim):
        for j in range(cdim):
            #assert shotNumber[i] != 0
            power = const2 ** -ADCBits[i]
            result[i,j] = matrix[i,j] * inputRange[i] * const1 *  power  / shotNumber[i]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def mHz ( np.ndarray [np.int64_t, ndim=2] matrix,
            np.ndarray[DTYPE_t, ndim=1] binWidth,
            np.ndarray[DTYPE_t, ndim=1] shotNumber):

    cdef int i,j
    cdef int const1 = 150
    cdef int rdim = matrix.shape[0]
    cdef int cdim = matrix.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.empty_like(matrix, dtype=DTYPE)

    for i in range(rdim):
        for j in range(cdim):
            result[i,j] = matrix[i,j] * ( const1 /  binWidth[i] ) / shotNumber[i]
    return matrix
