"""
This module reads ..cdf files for LC and GCMS
"""
import numpy as np
import netCDF4 as nc

from typing import Tuple


def read_gcms_cdf(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """ Reads GCMS ..cdf file. """
    gcms = nc.Dataset(filename)

    # get info
    mass = gcms['mass_values'][:].data
    intens = gcms['intensity_values'][:].data
    scantime = gcms['scan_acquisition_time'][:].data
    counts = gcms['point_count'][:].data
    ms_rng = round(gcms['mass_range_max'][:].data.max()) + 1

    # get data index
    m, = counts.shape
    n, = mass.shape
    tmp = np.ones(n)
    j = 0
    for i in range(m):
        if counts[i] > 0:
            tmp[j: j + counts[i]] = np.ones(counts[i]) * i * max_rng
            j += counts[i]

    # convert to data matrix with m scans x n mass bins
    x = np.zeros(m * ms_rng)
    tmp_ix = np.round(tmp + mass).astype(int) - 1
    x[tmp_ix] = intens
    xr = np.reshape(x, (m, max_rng))

    return scantime, xr


def read_lc_dad_cdf(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """ Reads LC-DAD chromatograms. """
    lc = nc.Dataset(filename)

    # intensity
    intens = lc["ordinate_values"][:].data

    # No. of records
    n = lc.dimensions["point_number"].size
    # record interval, in sec
    t = lc["actual_sampling_interval"][:].data
    # delayed time
    dt = lc["actual_delay_time"][:].data
    # retention time
    rt = np.arange(n) * t + dt + 0.1

    return rt, intens
