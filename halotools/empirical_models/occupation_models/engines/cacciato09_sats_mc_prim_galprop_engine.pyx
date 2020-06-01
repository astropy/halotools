# cython: language_level=2
""" Module containing the `~halotools.empirical_models.occupation_models.engines.cacciato09_sats_mc_prim_galprop_engine`
cython function driving the `mc_prim_galprop` function of the
`~halotools.empirical_models.occupation_models.Cacciato09Sats` class.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport pow, exp

__author__ = ('Johannes Ulf Lange', )
__all__ = ('cacciato09_sats_mc_prim_galprop_engine', )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cacciato09_sats_mc_prim_galprop_engine(mc_prim_galprop_in, randoms_in,
    alpha_sat_in, prim_galprop_cut_in, cnp.float64_t threshold):
    """
    Cython engine for determining Monte-Carlo realization of primary galaxy
    properties of satellites in the Cacciato09 CLF model. The function itself
    does not generate random numbers, it only converts input random numbers
    into primary galaxy properties. Generally, the function uses the input
    randoms to populate all zero entries of the ``mc_prim_galprop_in`` with
    according primary galaxy properties. It stops once all entries are non-zero
    or it runs out of randoms.

    Parameters
    ----------
    mc_prim_galprop_in : numpy.array
        Array storing Monte-Carlo realizations of primary galaxy properties.
        Values equal to zero signal a not-yet determined Monte-Carlo value.

    randoms_in : numpy.array
        Array storing random numbers in [0.0, 1.0) that the engine uses to
        populate ``mc_prim_galprop_in`` with primary galaxy properties.

    alpha_sat_in : numpy.array
        Array storing the pow-law slopes of the CLF.

    prim_galprop_cut_in : numpy.array
        Array storing the primary galaxy property cut-offs.

    threshold : float
        The lower limit on the primary galaxy properties that are assigned.

    Returns
    -------
    mc_prim_galprop : numpy.array
        Similar to ``mc_prim_galprop`` but with entries equal to zero populated
        with primary galaxy properties. If entries equal to zero are still
        present, the function must be called again with new randoms until all
        values are non-zero.
    """

    cdef cnp.float64_t[:] mc_prim_galprop = np.ascontiguousarray(
        mc_prim_galprop_in, dtype=np.float64)
    cdef cnp.float64_t[:] randoms = np.ascontiguousarray(randoms_in,
                                                         dtype=np.float64)
    cdef cnp.float64_t[:] alpha_sat = np.ascontiguousarray(alpha_sat_in,
                                                           dtype=np.float64)
    cdef cnp.float64_t[:] prim_galprop_cut = np.ascontiguousarray(
        prim_galprop_cut_in, dtype=np.float64)

    cdef cnp.float64_t prim_galprop_try, prim_galprop_max, p_accept

    cdef cnp.int64_t i_r= 0
    cdef cnp.int64_t n_r = len(randoms)

    cdef cnp.int64_t i_g = 0
    cdef cnp.int64_t n_g = len(mc_prim_galprop)

    cdef cnp.float64_t alpha_factor1, alpha_factor2

    while i_g < n_g and i_r < n_r:

        # Find the first "missing" entry in mc_prim_galprop.
        while i_g < n_g and mc_prim_galprop[i_g] != 0:
            i_g = i_g + 1

        if i_g == n_g:
            break

        # Draw a random primary galprop from a power-law. Because the integration
        # of the power-law could lead to infinities if alpha_sat > -1, we cut
        # at 1000 times the cut-off primary galaxy property. This should be
        # safe since  Phi_s(1000 * prim_galprop_cut) <= Phi_s(threshold) *
        # exp(-1000000). Also, we don't draw from power-laws with
        # alpha_sat > -1.1 directly because of the singularity at alpha_sat = -1
        # and possible numerical instabilities around it. Instead we at most
        # draw from a power-law with -1.1 and then reject certain points.

        prim_galprop_max = 1000. * prim_galprop_cut[i_g]
        if prim_galprop_max < 10. * threshold:
            prim_galprop_max = 10. * threshold

        alpha_factor1 = alpha_sat[i_g]
        if alpha_factor1 > -1.1:
            alpha_factor1 = -1.1
        prim_galprop_try = (pow(randoms[i_r] * (pow(prim_galprop_max / threshold,
            alpha_factor1 + 1.0) - 1.0) + 1.0,
            1.0 / (alpha_factor1 + 1.0)) * threshold)

        alpha_factor2 = -1.1 - alpha_sat[i_g]
        if alpha_factor2 > 0.:
            alpha_factor2 = 0.
        p_accept = (exp(- (prim_galprop_try*prim_galprop_try - threshold *
                           threshold) / (prim_galprop_cut[i_g]*
                                         prim_galprop_cut[i_g])) *
                        pow(prim_galprop_max / prim_galprop_try,
                            alpha_factor2))

        if randoms[i_r + 1] < p_accept:
            mc_prim_galprop[i_g] = prim_galprop_try

        i_r = i_r + 2

    return np.array(mc_prim_galprop)
