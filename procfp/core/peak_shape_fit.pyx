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
cdef double exp_erfc_ratio(double x):
    if x <= 0.:
        return exp(- x * x) / erfc(x)
    else:
        return exp(- x * x - approx_log_erfc(x))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void gradient_decent(double[::1] x, double[::1] y, double[::1] param):
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t m = param.shape[0]
        Py_ssize_t i
        int max_iter = 1000
        int niter = 0
        double * gd = <double *> malloc(m * sizeof(double))
        double * gdc = <double *> malloc(m * sizeof(double))
        double nd = <double> n
        double tol = 0.000001
        double r = 0.01
        double t0 = sqrt(2.)
        double t1 = sqrt(2. / M_PI)
        double nk = 0.
        double tx = 0.
        double err = 1.
        double a, u, s, b
        double z, zc, t, q, tc, sj, tq, s2, tu, tb, sg, sd, td

    for i in range(n):
        nk += y[i]
        tx += x[i] * y[i]

    print(np.asarray(param))

    while err <= tol or niter < max_iter:
        a = param[0]
        u = param[1]
        s = param[2]
        b = param[3]

        tc = t0 * s
        s2 = s * s
        sj = b * s2
        zc = u + sj
        tq = 0.
        tu = 0.
        tb = 0.
        for i in range(n):
            z = (zc - x[i]) / tc
            q = exp_erfc_ratio(z) * y[i] * t1
            tb += q * s
            tq += q * (sj - u + x[i]) / s2
            tu += q / s

        gdc[0] = nk / a
        gdc[1] = nk * b - tu
        gdc[2] = nk * b * b * s - tq
        gdc[3] = nk / b + nk * u + nk * sj - tx - tb

        # print(nk, gdc[0], gdc[1], gdc[2], gdc[3])
        if np.isnan(gdc[0]):
            break

        if niter >= 1:
            sg = 0.
            sd = 0.
            tc = 0.
            td = 0.
            for i in range(m):
                t = gd[i] - gdc[i]
                sd += t * t
                sg += t * gd[i]
                gd[i] = gdc[i]
                tc += gdc[i] * gdc[i]
                td += gd[i] * gd[i]
            # r = r * fabs(sg) / sd
            # print(r, sg, sd)
            err = sqrt(sd) / max(sqrt(tc), sqrt(td))

        for i in range(m):
            param[i] += r * gdc[i]

        niter += 1

    free(gd)
    free(gdc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void emg(double[::1] x, double[::1] param, double[::1] y):
    """ Fits EMG """
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void estimate_emg(double[::1] x, double[::1] y, double[::1] param):
    """ Estimates parameters of EMG. """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double nd = <double> n
        double m3 = 0.
        double s = 0.
        double nk = 0.
        double m = 0.
        double a = 0.
        double b, g, tau, t


    for i in range(n):
        nk += y[i]
        m += x[i] * y[i]
    m /= nk

    for i in range(1, n):
        a += (y[i - 1] + y[i]) * (x[i] - x[i - 1]) / 2.

    for i in range(n):
        b = x[i] - m
        t = b * b * y[i]
        s += t
        m3 += b * t
    m3 /= nk
    s = sqrt(s / (nk - 1.))
    g = m3 / pow(s, 3.)

    # estimated parameters
    t = pow(g / 2., 1. / 3.)
    tau = s * t
    param[0] = a
    param[1] = m - tau
    param[2] = s * sqrt(1. - t * t)
    param[3] = 1. / tau
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
