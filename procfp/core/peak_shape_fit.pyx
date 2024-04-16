cimport cython

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, M_PI

import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64


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
