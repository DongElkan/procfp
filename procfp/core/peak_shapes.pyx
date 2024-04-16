cimport cython

from libc.math cimport sqrt, log, exp, erfc, M_PI


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
    """ Approximates log erfc: 1. - erf when x > 0. """
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
cdef double exp_erfc_mul(double x, double z):
    if z <= 0.:
        return exp(-x * x) * erfc(z)
    return exp(- x * x + approx_log_erfc(z))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void emg(double[::1] x, double[::1] param, double[::1] y):
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
        double h = param[0]
        double mu = param[1]
        double s = param[2]
        double t = param[3]
        double gc = h * s / t * sqrt(M_PI / 2.)
        double ec = mu / t + s * s / (2. * t * t)
        double st = s / t
        double zc = (mu / s + st) / sqrt(2.)
        double c2 = sqrt(2.) * s
        double m, z, g

    for i in range(n):
        m = (x[i] - mu) / s
        z = zc - x[i] / c2
        if z < 0.:
            y[i] = gc * exp(ec - x[i] / t) * erfc(z)
        else:
            g = approx_log_erfc(z) + m
            y[i] = lk * exp(g) * a
