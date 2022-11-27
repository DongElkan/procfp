cimport cython

from libc.math cimport sqrt, pow, exp, log, erfc, M_PI

import numpy as np
cimport numpy as np

np.import_array()


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
cdef double approx_log_erfc(double x):
    """ Approximates log erfc: 1. - erf when x >= 0. """
    cdef:
        double t = 1. / (1. + 0.5 * x)
        double t0 = 1.
        double v = 0.

    for i in range(9):
        t0 *= t
        v += COEF_ERF[i] * t0
    v += COEF_ERF[9]

    return log(t) - x * x + v


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void emg(double[::1] x, double mu, double sigma, double tau, double h, double[::1] y):
    """ Fits EMG """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double c = sigma / tau
        double sq2 = sqrt(2.)
        double lk = log(h * c * sqrt(M_PI / 2.))
        double m, z, e, s, g

    for i in range(n):
        m = (x[i] - mu) / sigma
        z = (c - m) / sq2
        if z < 0.:
            s = c * c / 2. - (m * c) + lk + erfc(z)
        elif z > 6.71e7:
            # to avoid overflow
            s = log(h / (1. - m / c)) - m * m / 2.
        else:
            s = lk + approx_log_erfc + z * z - m * m / 2
        y[i] = exp(s)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef (double, double, double) estimate_emg(double[::1] x):
    """
    Estimates parameters of EMG.

    Args:
        x: The peak for fitting

    Returns:
        Parameters

    References:
        [1] wiki/Exponentially_modified_Gaussian_distribution
        [2] Kalambet Y, et al. J Chemometrics. 2011; 25, 352–356.
        [3] Jeansonne MS, et al. J Chromatogr Sci. 1991, 29, 258-266.

    """
    cdef:
        Py_ssize_t n = x.shape[0]
        Py_ssize_t i
        double nd = <double> n
        double m3 = 0.
        double s = 0.
        double t = 0.
        double m, a, b, u, g

    for i in range(n):
        t += x[i]

    m = t / nd
    for i in range(n):
        a = x[i] - m
        s += a * a
        m3 += a * a * a

    m3 /= nd
    s = sqrt(s / (n - 1.))
    g = m3 / pow(s, 3.)

    # estimated parameters
    a = pow(g / 2., 1. / 3.)
    t = s * a
    u = m - t
    s = s * sqrt(1. - a * a)

    return u, s, t


def fit_curve(rt: np.ndarray, intensity: np.ndarray, shape: str = 'emg')\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the curve using scipy's curve_fit optimization.

    Args:
        rt: Retention time.
        intensity: Peak intensities.
        shape: Function for fitting the peak. {`emg`, `gaussian`}.
            emg: exponentially modified gaussian function
            gaussian: gaussian function
            Defaults to `emg`.

    Returns:
        np.ndarray: Fitted peaks.
        np.ndarray: Parameters for fitting the peak.

    """
    # initialization
    j = np.argmax(intensity)
    h0, m0 = intensity[j], rt[j]
    s0 = np.std(rt)
    t0 = 1

    # fit curve using optimization
    if shape == 'emg':
        param, _ = optimize.curve_fit(_emg, rt, intensity, p0=(h0, s0, m0, t0))
        return _emg(rt, *param), param

    if shape == 'gaussian':
        param, _ = optimize.curve_fit(_gaussian, rt, intensity, p0=(h0, m0, s0))
        return _gaussian(rt, *param), param

    raise ValueError("Unrecognized shape for fitting. Expected `emg` or "
                     f"`gaussian`, got {shape}.")


def get_peak_param(rt: np.ndarray, intensity: np.ndarray)\
        -> Tuple[float, float, float]:
    """
    Get peak parameters, including peak width at half height,
    peak width at base and peak height.

    Args:
        rt: Retention time.
        intensity: Peak intensities.

    Returns:
        tuple: A tuple of peak height, peak width at half height,
            peak width at base.

    """
    h = intensity.max()
    # half peak height
    h2 = h/2
    # approximated peak width at half height
    rt_h2 = rt[intensity >= h2]
    wh = rt_h2.max() - rt_h2.min()
    # approximated peak width at base
    rt_base = rt[intensity >= h * 0.001]
    wb = rt_base.max() - rt_base.min()
    return h, wh, wb

