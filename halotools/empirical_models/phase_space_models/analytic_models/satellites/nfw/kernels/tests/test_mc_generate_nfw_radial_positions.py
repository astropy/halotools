"""
"""
import numpy as np
import pytest

from ..mc_generate_nfw_radial_positions import mc_generate_nfw_radial_positions
from ........custom_exceptions import HalotoolsError

__all__ = ('test_mc_generate_nfw_radial_positions1', )


def test_mc_generate_nfw_radial_positions1():
    with pytest.raises(HalotoolsError) as err:
        __ = mc_generate_nfw_radial_positions(num_pts=10)


def test_mc_generate_nfw_radial_positions2():

    with pytest.raises(HalotoolsError) as err:
        __ = mc_generate_nfw_radial_positions(num_pts=10, halo_mass=np.logspace(12, 13, 5))


def test_mc_generate_nfw_radial_positions3():

    with pytest.raises(HalotoolsError) as err:
        __ = mc_generate_nfw_radial_positions(num_pts=10, halo_radius=np.linspace(2, 3, 5))


def test_mc_generate_nfw_radial_positions4():

    with pytest.raises(HalotoolsError) as err:
        __ = mc_generate_nfw_radial_positions(num_pts=10, halo_radius=0.5, conc=np.linspace(2, 3, 5))
