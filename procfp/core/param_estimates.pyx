cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, pow, exp, log, erfc, M_PI, fabs

import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64


cdef double COEF_ERF[10]
COEF_ERF[0] = 1.00002368
COEF_ERF[1] = 0.37409196
COEF_ERF[2] = 0.09678418
COEF_ERF[3] = -0.18628806
COEF_ERF[4] = 0.27886807
COEF_ERF[5] = -1.13520398
COEF_ERF[6] = 1.48851587
COEF_ERF[7] = -0.82215223
COEF_ERF[8] = 0.17087277
COEF_ERF[9] = -1.26551223  # constant c0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double approx_log_erfc(double x):
    """ Approximates log erfc(x): 1. - erf(x) when x > 0. """
    cdef:
        double t = 1. / (1. + 0.5 * x)
        double t0 = 1.
        double v = 0.

    for i in range(9):
        t0 *= t
        v += COEF_ERF[i] * t0
    v += COEF_ERF[9]

    return log(t) - x * x + v


@cython.cdivision(True)
cdef double exp_erfc_ratio(double x):
    if x <= 0.:
        return exp(- x * x) / erfc(x)
    else:
        return exp(- x * x - approx_log_erfc(x))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void estimate_emg(double[::1] x, double[::1] y, double[::1] param):
    """
    Estimates parameters of EMG.

    References:
        Kalambet Y, et al. Reconstruction of chromatographic peaks
        using the exponentially modified Gaussian function.
        J Chemometrics. 2011, 25, 352–356.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double nd = <double> n
        double a = 0.
        double a2 = 0.
        double h = y[0]
        double b, m, s

    for i in range(n):
        if y[i] > h:
            m = x[i]
            h = y[i]
        a += x[i]
        a2 += x[i] * x[i]
    s = sqrt((a2 - (a * a) / nd) / (nd - 1.))

    param[0] = h
    param[1] = m
    param[2] = s
    param[3] = 1.
    print(np.asarray(param))


cpdef fit_peak(double[::1] rt, double[::1] peak):
    """
    Fit the curve using scipy's curve_fit optimization.

    Args:
        rt: Retention time.
        peak: Peak intensities.

    """
    cdef:
        Py_ssize_t n = peak.shape[0]
        Py_ssize_t i
        double[::1] param = np.zeros(4, dtype=DTYPE)

    estimate_emg(rt, peak, param)
    gradient_decent(rt, peak, param)

    return param


cpdef construct_peak(double[::1] x, double[::1] param):
    """
    Constructs EMG peak.

    Args:
        x: Retention times.
        param: Parameters fitted previously, size of 4.
            param[0]: peak area
            param[1]: EMG center, mu
            param[2]: standard deviation, sigma
            param[3]: lambda, the exponent relaxation time is thus
                calculated by 1/lambda.

    Returns:
        EMG peak.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        double[::1] y = np.zeros(n, dtype=DTYPE)

    emg(x, param, y)

    return np.asarray(y)
