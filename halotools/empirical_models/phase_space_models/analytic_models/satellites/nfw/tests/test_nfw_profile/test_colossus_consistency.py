"""
"""
import numpy as np
from astropy.cosmology import Planck15
from itertools import product

from ...nfw_profile import NFWProfile


def test_colossus_vmax_consistency_z0():
    """
    """
    try:
        from colossus.cosmology import cosmology
        from colossus.halo.profile_nfw import NFWProfile as ColossusNFW
    except ImportError:
        return
    _cosmo = cosmology.setCosmology('planck15')
    nfw = NFWProfile(cosmology=Planck15)

    masses_to_check = 10**np.array((11., 12, 13, 14, 15))
    conc_to_check = [5., 10., 15.]

    z = 0
    for M, c in product(masses_to_check, conc_to_check):

        colossus_nfw = ColossusNFW(M=M, c=c, mdef='vir', z=z)
        colossus_vmax, colossus_rmax = colossus_nfw.Vmax()
        halotools_vmax = nfw.vmax(M, c)[0]

        msg = ("\nM = {0:.2e}, c = {1:.1f}, z = {2:.1f}\n"
            "Colossus Vmax = {3:.2f}\nHalotools Vmax = {4:.2f}\n".format(
                M, c, z, colossus_vmax, halotools_vmax))
        assert np.allclose(colossus_vmax, halotools_vmax, rtol=0.05), msg


def test_colossus_vmax_consistency_z1():
    """
    """
    try:
        from colossus.cosmology import cosmology
        from colossus.halo.profile_nfw import NFWProfile as ColossusNFW
    except ImportError:
        return
    _cosmo = cosmology.setCosmology('planck15')

    masses_to_check = 10**np.array((11., 12, 13, 14, 15))
    conc_to_check = [5., 10., 15.]

    z = 1.
    nfw = NFWProfile(cosmology=Planck15, redshift=z)

    for M, c in product(masses_to_check, conc_to_check):

        colossus_nfw = ColossusNFW(M=M, c=c, mdef='vir', z=z)
        colossus_vmax, colossus_rmax = colossus_nfw.Vmax()
        halotools_vmax = nfw.vmax(M, c)[0]

        msg = ("\nM = {0:.2e}, c = {1:.1f}, z = {2:.1f}\n"
            "Colossus Vmax = {3:.2f}\nHalotools Vmax = {4:.2f}\n".format(
                M, c, z, colossus_vmax, halotools_vmax))
        assert np.allclose(colossus_vmax, halotools_vmax, rtol=0.05), msg

