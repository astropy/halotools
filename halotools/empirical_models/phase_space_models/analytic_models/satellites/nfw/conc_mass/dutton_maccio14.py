"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np


__all__ = ('dutton_maccio14', )


def dutton_maccio14(mass, redshift):
    r""" Power-law fit to the concentration-mass relation from
    Equations 12 & 13 of Dutton and Maccio 2014, arXiv:1402.7073.

    :math:`\log_{10}c(M, z) \equiv a + b\log_{10}(M / M_{0}),`

    where :math:`a, b, M_{0}` are defined as follows:

    :math:`a = 0.537 + (1.025 - 0.537)\exp(-0.718z^{1.08})`

    :math:`b = -0.097 + 0.024z`

    :math:`M_{0} = 10^{12}M_{\odot}/h`

    Parameters
    ----------
    mass : array_like

    redshift : array_like

    Returns
    -------
    concentration : array_like

    Notes
    -----
    This model is based on virial mass definition and
    was only calibrated for the Planck 1-year cosmology.

    Examples
    --------
    >>> c = dutton_maccio14(1e12, 0)
    >>> c = dutton_maccio14(np.logspace(11, 15, 100), 0)
    """

    a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * redshift**1.08)
    b = -0.097 + 0.024 * redshift
    m0 = 1.e12

    logc = a + b * np.log10(mass / m0)
    return 10**logc
