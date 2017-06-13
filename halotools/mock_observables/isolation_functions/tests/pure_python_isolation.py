"""
"""
import numpy as np


def naive_spherical_isolation(x1arr, y1arr, z1arr, x2arr, y2arr, z2arr,
        rmax, xperiod, yperiod, zperiod):
    """
    """
    npts1 = len(x1arr)
    is_isolated = np.zeros(npts1, dtype=bool)
    for i, x1, y1, z1 in zip(range(npts1), x1arr, y1arr, z1arr):
        dx = np.abs(x2arr - x1)
        dy = np.abs(y2arr - y1)
        dz = np.abs(z2arr - z1)
        dx = np.where(dx > xperiod/2., xperiod - dx, dx)
        dy = np.where(dy > yperiod/2., yperiod - dy, dy)
        dz = np.where(dz > zperiod/2., zperiod - dz, dz)
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        if np.any(d < rmax):
            is_isolated[i] = False
        else:
            is_isolated[i] = True
    return is_isolated


def naive_cylindrical_isolation(x1arr, y1arr, z1arr, x2arr, y2arr, z2arr,
        rp_max, pi_max, xperiod, yperiod, zperiod):
    """
    """
    npts1 = len(x1arr)
    is_isolated = np.zeros(npts1, dtype=bool)
    for i, x1, y1, z1 in zip(range(npts1), x1arr, y1arr, z1arr):
        dx = np.abs(x2arr - x1)
        dy = np.abs(y2arr - y1)
        dz = np.abs(z2arr - z1)
        dx = np.where(dx > xperiod/2., xperiod - dx, dx)
        dy = np.where(dy > yperiod/2., yperiod - dy, dy)
        dz = np.where(dz > zperiod/2., zperiod - dz, dz)
        dxy = np.sqrt(dx**2 + dy**2)

        if np.any((dxy < rp_max) & (dz < pi_max)):
            is_isolated[i] = False
        else:
            is_isolated[i] = True
    return is_isolated
