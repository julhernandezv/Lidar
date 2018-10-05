


import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


# """ Function to calculate range_corrected from Lidar data"""
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def cy_range_corrected( np.ndarray [DTYPE_t, ndim=2] matrix,
                    np.ndarray [DTYPE_t, ndim=1] rang):
    cdef int i,j
    cdef int rdim = matrix.shape[0]
    cdef int cdim = matrix.shape[1]
    # cdef np.ndarray[DTYPE_t, ndim=2] result = np.empty_like(matrix, dtype=DTYPE)

    for i in range(rdim):
        for j in range(cdim):
            # result [i,j] = matrix[i,j] * rang[j]
            matrix[i,j] = matrix[i,j] ** rang[j]
    return matrix

# """ Function to calculate mVolts from Lidar raw data"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def cy_mVolts ( np.ndarray [DTYPE_t, ndim=2] matrix,
            np.ndarray[DTYPE_t, ndim=1] inputRange,
            np.ndarray[DTYPE_t, ndim=1] ADCBits,
            np.ndarray[DTYPE_t, ndim=1] shotNumber):

    cdef int i,j
    cdef DTYPE_t power
    cdef int const1 = 1000
    cdef int const2 = 2
    cdef int rdim = matrix.shape[0]
    cdef int cdim = matrix.shape[1]
    # cdef np.ndarray[DTYPE_t, ndim=2] result = np.empty_like(matrix, dtype=DTYPE)

    for i in range(rdim):
        for j in range(cdim):
            #assert shotNumber[i] != 0
            power = const2 ** -ADCBits[i]
            # print  power
            # result[i,j] = matrix[i,j] * inputRange[i] * const1 *  power  / shotNumber[i]
            matrix[i,j] *= inputRange[i] * const1 *  power  / shotNumber[i]
    return matrix



# """ Function to calculate mHz from Lidar raw data"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def cy_mHz ( np.ndarray [DTYPE_t, ndim=2] matrix,
            np.ndarray[DTYPE_t, ndim=1] binWidth,
            np.ndarray[DTYPE_t, ndim=1] shotNumber):

    cdef int i,j
    cdef int const1 = 150
    cdef int rdim = matrix.shape[0]
    cdef int cdim = matrix.shape[1]
    #cdef np.ndarray[DTYPE_t, ndim=2] result = np.empty_like(matrix, dtype=DTYPE)

    for i in range(rdim):
        for j in range(cdim):
            matrix[i,j] *=  ( const1 /  binWidth[i] ) / shotNumber[i]
    return matrix


# """ Function to substrac  Lidar's background"""
@cython.boundscheck(False)
@cython.wraparound(False)
def cy_brackground ( np.ndarray [DTYPE_t, ndim=2] matrix,
            np.ndarray[DTYPE_t, ndim=2] bkg,
            np.ndarray[np.int16_t, ndim=1] labelA,
            np.ndarray[np.int16_t, ndim=1] labelP,
            int lenP ):

    cdef int i,j,c
    cdef int rdim = matrix.shape[0]
    cdef int cdim = matrix.shape[1]
    cdef int bkgdim = bkg.shape[1] - 1
    #cdef np.ndarray[DTYPE_t, ndim=2] result = np.empty_like(matrix, dtype=DTYPE)

    for i in range(rdim):
        for j in range(cdim):
            c = labelP[j] + labelA[j] * lenP
            matrix[i,j] -= bkg[i,c]

    return matrix


# """ Function to derived by centered difference method"""
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def cy_derived ( np.ndarray [DTYPE_t, ndim=2] matrix,
#             D delta):
