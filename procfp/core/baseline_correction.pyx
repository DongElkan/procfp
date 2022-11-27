cimport cython

from libc.math cimport exp, fabs, sqrt
from libc.stdlib cimport malloc, free, calloc

import numpy as np
cimport numpy as np

from numpy import linalg

DTYPE = np.float64


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double adaptive_reweight(double[::1] w, double[::1] d, double t):
    cdef:
        Py_ssize_t n = d.shape[0]
        Py_ssize_t i, j
        Py_ssize_t nd = 0
        int * neg_ix = <int *> malloc(n * sizeof(int))
        double * neg_d = <double *> malloc(n * sizeof(double))
        double tot_nd = 0.
        double m_nd

    for i in range(n):
        if d[i] >= 0.:
            w[i] = 0.
        else:
            neg_d[nd] = d[i]
            tot_nd += d[i]
            neg_ix[nd] = <int> i
            nd += 1

    # re-weight current baseline
    m_nd = neg_d[0]
    for i in range(nd):
        j = neg_ix[i]
        w[j] = exp(t * neg_d[i] / tot_nd)
        if neg_d[i] > m_nd:
            m_nd = neg_d[i]

    # compensation for start and end
    w[0] = exp(t * m_nd / tot_nd)
    w[n - 1] = w[0]

    free(neg_ix)
    free(neg_d)

    return fabs(tot_nd)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double asymmetric_reweight(double[::1] d, double[::1] w):
    cdef:
        Py_ssize_t n = d.shape[0]
        Py_ssize_t i, j
        int nd = 0
        double tot_neg = 0.
        double tot_neg2 = 0.
        double err = 0.
        double m, s, f, ndt, t

    for i in range(n):
        if d[i] < 0.:
            nd += 1
            tot_neg += d[i]
            tot_neg2 += d[i] * d[i]

    if nd > 0:
        ndt = <double> nd
        m = tot_neg / ndt
        s = sqrt((tot_neg2 - ndt * m * m) / (ndt - 1.))
        for i in range(n):
            t = w[i]
            if d[i] >= 0.:
                f = 2. * (d[i] - (2. * s - m)) / s
                if f > 60.:
                    f = 60.
                w[i] = 1. / (1. + exp(f))
            else:
                w[i] = 1.
            err += (w[i] - t) * (w[i] - t)
    else:
        for i in range(n):
            err += w[i] * w[i]

    return sqrt(err)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void smoothing_matrix(int n, int d, double c, double[:, ::1] a, double[:, ::1] b):
    """
    Generates smoothing matrix from identify matrix with derivative d.
    
    """
    cdef:
        Py_ssize_t i, j, k, j0, j1, j2
        int n2 = n - d
        int d2 = 2 * d - 1
        double * id_col = <double *> calloc(n, sizeof(double))
        double s

    for i in range(n):
        id_col[i] = 1.
        j1 = min(i + 1, n - 1)
        for j in range(1, d + 1):
            j0 = max(0, i - j)
            for k in range(j0, j1):
                id_col[k] = id_col[k + 1] - id_col[k]

        j2 = min(j1, n2)
        for j in range(j0, j2):
            a[j, i] = id_col[j]
        for j in range(j0, j1 + 1):
            id_col[j] = 0.

    for i in range(n):
        j0 = max(0, i - d2 + 1)
        j1 = min(i + d + 1, n)
        for j in range(j0, j1):
            s = 0.
            for k in range(j0, min(j + d, n2)):
                s += a[i, k] * a[k, j]
            b[i, j] = s * c

    free(id_col)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void smooth_matrix_weight(double[:, ::1] b, double[::1] w, double[:, ::1] c, bint only_diag):
    """ Updates smoothing matrix. """
    cdef:
        Py_ssize_t n = b.shape[0]
        Py_ssize_t i, j

    if not only_diag:
        for i in range(n):
            for j in range(n):
                c[i, j] = b[i, j]
            c[i, i] += w[i]
    else:
        for i in range(n):
            c[i, i] = b[i, i] + w[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double[::1] airpls(double[::1] x, double lambda_):
    """
    Corrects baseline using adaptive iteratively re-weighted penalized
    least squares.
    
    Args:
        x: Curve
        lambda_: Penalty

    Returns:
        Corrected baseline
    
    References:
        [1] Zhang ZM, Chen S, Liang YZ. Baseline correction using
            adaptive iteratively reweighted penalized least squares.
            Analyst. 2010, 135, 1138-1146.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i, j
        int max_iter = 100
        double[:, ::1] a = np.zeros((n - 2, n), dtype=DTYPE)
        double[:, ::1] b = np.zeros((n, n), dtype=DTYPE)
        double[:, ::1] c = np.zeros((n, n), dtype=DTYPE)
        double[::1] w = np.ones(n, dtype=DTYPE)
        double[::1] wx = np.zeros(n, dtype=DTYPE)
        double[::1] d = np.zeros(n, dtype=DTYPE)
        double[::1] z
        double tol = 0.001
        double s = 0.
        double dd

    for i in range(n):
        s += fabs(x[i])
    tol *= s

    # smoothing matrix
    smoothing_matrix(n, 2, lambda_, a, b)
    smooth_matrix_weight(b, w, c, 0)
    for i in range(max_iter):
        # solve background using least squares
        for j in range(n):
            wx[j] = w[i] * x[i]
        z = linalg.solve(c, wx)
        # remove peak signal from the background for next iteration
        for j in range(n):
            d[j] = x[j] - z[j]

        s = adaptive_reweight(w, d, i + 1.)
        if s <= tol:
            break

        smooth_matrix_weight(b, w, c, 1)

    return z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef arpls(double[::1] x, double lambda_):
    """
    Corrects baseline using asymmetrically reweighted penalized least
    squares.

    Args:
        x: Curve
        lambda_: Penalty.

    Returns:
        array: Fitted baseline.

    References:
        [1] Baek SJ, Park A, Ahn YJ, Choo J. Baseline correction using
            asymmetrically re-weighted penalized least squares
            smoothing. Analyst. 2015, 140, 250-257.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i, j
        int max_iter = 100
        double[:, ::1] a = np.zeros((n - 2, n), dtype=DTYPE)
        double[:, ::1] b = np.zeros((n, n), dtype=DTYPE)
        double[:, ::1] c = np.zeros((n, n), dtype=DTYPE)
        double[::1] w = np.ones(n, dtype=DTYPE)
        double[::1] wx = np.zeros(n, dtype=DTYPE)
        double[::1] d = np.zeros(n, dtype=DTYPE)
        double[::1] z
        double tol = 0.000001
        double s0, s

    # smoothing matrix
    smoothing_matrix(n, 2, lambda_, a, b)
    smooth_matrix_weight(b, w, c, 0)
    for i in range(max_iter):
        # solve background using least squares
        s0 = 0.
        for j in range(n):
            wx[j] = w[i] * x[i]
            s0 += w[i] * w[i]

        z = linalg.solve(c, wx)
        # remove peak signal from the background for next iteration
        for j in range(n):
            d[j] = x[j] - z[j]

        s = asymmetric_reweight(w, d)
        if s / s0 <= tol:
            break

        smooth_matrix_weight(b, w, c, 1)

    return z
