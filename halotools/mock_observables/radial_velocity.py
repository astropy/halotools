"""
"""
import numpy as np


def signed_dx(xs, xc, xperiod):
    """
    """
    dx_uncorrected = xs - xc
    abs_dx_uncorrected = np.abs(dx_uncorrected)
    pbc_flip_mask = abs_dx_uncorrected > xperiod/2.
    pbc_flip_sign = np.where(pbc_flip_mask, -1, 1)
    original_sign = np.sign(dx_uncorrected)
    abs_dx = np.where(pbc_flip_mask, np.abs(abs_dx_uncorrected - xperiod), abs_dx_uncorrected)
    result = pbc_flip_sign*original_sign*abs_dx
    return np.atleast_1d(result)


def radial_distance_and_velocity(xs, ys, zs, vxs, vys, vzs, xc, yc, zc, vxc, vyc, vzc, period):
    """
    """
    npts = len(np.atleast_1d(xs))
    vrad = np.zeros(npts)

    try:
        xperiod, yperiod, zperiod = period
    except TypeError:
        xperiod, yperiod, zperiod = period, period, period

    dx = signed_dx(xs, xc, xperiod)
    dy = signed_dx(ys, yc, yperiod)
    dz = signed_dx(zs, zc, zperiod)
    drad = np.sqrt(dx**2 + dy**2 + dz**2)

    dvx = np.atleast_1d(vxs - vxc)
    dvy = np.atleast_1d(vys - vyc)
    dvz = np.atleast_1d(vzs - vzc)

    idx_nonzero_distance = drad > 0
    num_nonzero_distance = np.count_nonzero(idx_nonzero_distance)
    num_zero_distance = npts - num_nonzero_distance

    if num_zero_distance > 0:
        term1a = dvx[~idx_nonzero_distance]**2
        term2a = dvy[~idx_nonzero_distance]**2
        term3a = dvz[~idx_nonzero_distance]**2
        vrad[~idx_nonzero_distance] = np.sqrt(term1a + term2a + term3a)

    if num_nonzero_distance > 0:
        term1b = dx[idx_nonzero_distance]*dvx[idx_nonzero_distance]
        term2b = dy[idx_nonzero_distance]*dvy[idx_nonzero_distance]
        term3b = dz[idx_nonzero_distance]*dvz[idx_nonzero_distance]
        numerator = term1b + term2b + term3b
        denominator = drad[idx_nonzero_distance]
        vrad[idx_nonzero_distance] = numerator/denominator

    return drad, vrad

