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
cdef double exp_erfc_ratio(double x, double z):
    if z <= 0.:
        return exp(- x * x) / erfc(z)
    else:
        return exp(- x * x - approx_log_erfc(z))


@cython.cdivision(True)
cdef double exp_erfc_mul(double x, double z):
    if z <= 0.:
        return exp(-x * x) * erfc(z)
    return exp(- x * x + approx_log_erfc(z))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void emg(double[::1] x, double[::1] param, double[::1] y):
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double a = param[0]
        double mu = param[1]
        double s = param[2]
        double lb = param[3]
        double lk = lb / 2.
        double c1 = lb * mu + lb * lb * s * s / 2.
        double c2 = mu + lb * s * s
        double sb = sqrt(2.) * s
        double m, z, g

    for i in range(n):
        m = c1 - lb * x[i]
        z = (c2 - x[i]) / sb
        if z < 0.:
            y[i] = lk * exp(m) * erfc(z) * a
        else:
            g = approx_log_erfc(z) + m
            y[i] = lk * exp(g) * a
