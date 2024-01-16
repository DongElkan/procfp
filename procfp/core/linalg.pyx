cimport cython

from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void ldl(double[:, ::1] x, double[:, ::1] w, double[::1] d):
    """
    Calculates LDL decomposition of symmetric matrix:
        x = w * d * w'.
        
    Notes:
        The lower unit triangular matrix is set in advance (i.e., in
        function that calls `ldl` to avoid pre-allocation here). The
        matrix w is size of n by n, and d is a vector of size n, where
        n is the number of rows in x.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t c = x.shape[1]
        Py_ssize_t i, j, k
        double s, t, a

    for i in range(n):
        t = 0.
        for j in range(i):
            s = 0.
            for k in range(j):
                s += w[i, k] * w[j, k] * d[k]
            a = (x[i, j] - s) / d[j]
            w[i, j] = a
            t += a * a * d[j]
        d[i] = x[i, i] - t
        w[i, i] = 1.


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void solve_linear(double[:, ::1] ml, double[::1] d, double[::1] z, double[::1] x):
    """
    Solves linear equation system by LDL decomposition.
    For linear system Ax = z, and LDL decomposition of A:
        A = DLD'
    Solve the equations sequentially:
        Lu = z
        Dv = u
        L'x = v

    Args:
        ml: Lower triangular matrix L.
        d: Diagonal of matrix D, 1D array.
        z: Dependent variable.
        x: Solution

    """
    cdef:
        Py_ssize_t n = d.shape[0]
        Py_ssize_t i, j, k
        double * u = <double *> malloc(n * sizeof(double))
        double s = 0.

    # solve Lu = z
    for i in range(n):
        s = 0.
        for j in range(i):
            s += ml[i, j] * u[j]
        u[i] = z[i] - s

    # solve L'Dx = u
    for i in range(1, n + 1):
        j = n - i
        s = 0.
        for k in range(j + 1, n):
            s += ml[k, j] * x[k]
        x[j] = u[j] / d[j] - s

    free(u)


cpdef cal_ldl(double[:, ::1] x):
    cdef:
        Py_ssize_t n = x.shape[0]
        double[:, ::1] w = np.zeros((n, n), dtype=DTYPE)
        double[::1] d = np.zeros(n, dtype=DTYPE)

    ldl(x, w, d)
    return w, d


cpdef solve_x(double[:, ::1] a, double[::1] b):
    cdef:
        Py_ssize_t n = b.shape[0]
        double[:, ::1] w = np.zeros((n, n), dtype=DTYPE)
        double[::1] d = np.zeros(n, dtype=DTYPE)
        double[::1] x = np.zeros(n, dtype=DTYPE)

    ldl(a, w, d)
    solve_linear(w, d, b, x)
    return x
