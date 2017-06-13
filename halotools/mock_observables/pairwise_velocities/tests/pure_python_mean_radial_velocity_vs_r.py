""" Module storing brute force python function `pure_python_mean_radial_velocity_vs_r`
used for unit-testing of the `mean_radial_velocity_vs_r` Cython implementation.
"""
import numpy as np


__all__ = ('pure_python_mean_radial_velocity_vs_r', )


def pure_python_mean_radial_velocity_vs_r(
        sample1, velocities1, sample2, velocities2, rmin, rmax, Lbox=None):
    """ Brute force pure python function calculating mean radial velocities
    in a single bin of separation.
    """
    if Lbox is None:
        xperiod, yperiod, zperiod = np.inf, np.inf, np.inf
    else:
        xperiod, yperiod, zperiod = Lbox, Lbox, Lbox

    npts1, npts2 = len(sample1), len(sample2)

    running_tally = []
    for i in range(npts1):
        for j in range(npts2):
            dx = sample1[i, 0] - sample2[j, 0]
            dy = sample1[i, 1] - sample2[j, 1]
            dz = sample1[i, 2] - sample2[j, 2]
            dvx = velocities1[i, 0] - velocities2[j, 0]
            dvy = velocities1[i, 1] - velocities2[j, 1]
            dvz = velocities1[i, 2] - velocities2[j, 2]

            xsign_flip, ysign_flip, zsign_flip = 1, 1, 1
            if dx > xperiod/2.:
                dx = xperiod - dx
                xsign_flip = -1
            elif dx < -xperiod/2.:
                dx = -(xperiod + dx)
                xsign_flip = -1

            if dy > yperiod/2.:
                dy = yperiod - dy
                ysign_flip = -1
            elif dy < -yperiod/2.:
                dy = -(yperiod + dy)
                ysign_flip = -1

            if dz > zperiod/2.:
                dz = zperiod - dz
                zsign_flip = -1
            elif dz < -zperiod/2.:
                dz = -(zperiod + dz)
                zsign_flip = -1

            d = np.sqrt(dx*dx + dy*dy + dz*dz)

            if (d > rmin) & (d < rmax):
                vrad = (dx*dvx*xsign_flip + dy*dvy*ysign_flip + dz*dvz*zsign_flip)/d
                running_tally.append(vrad)

    if len(running_tally) > 0:
        return np.mean(running_tally)
    else:
        return 0.
