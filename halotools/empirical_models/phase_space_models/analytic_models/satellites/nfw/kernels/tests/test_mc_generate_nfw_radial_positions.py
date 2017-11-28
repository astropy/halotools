"""
"""
import pytest

from ..mc_generate_nfw_radial_positions import mc_generate_nfw_radial_positions
from ........custom_exceptions import HalotoolsError

__all__ = ('test_mc_generate_nfw_radial_positions1', )


def test_mc_generate_nfw_radial_positions1():
    with pytest.raises(HalotoolsError) as err:
        __ = mc_generate_nfw_radial_positions(num_pts=10)
    substr = ("If keyword argument ``halo_radius`` is unspecified, "
        "argument ``halo_mass`` must be specified.")
    assert substr in err.value.args[0]


def test_mc_generate_nfw_radial_positions2():

    with pytest.raises(HalotoolsError) as err:
        __ = mc_generate_nfw_radial_positions(num_pts=10, halo_mass=(2, 3))
    substr = ("Input ``halo_mass`` must be a float")
    assert substr in err.value.args[0]


def test_mc_generate_nfw_radial_positions3():

    with pytest.raises(HalotoolsError) as err:
        __ = mc_generate_nfw_radial_positions(num_pts=10, halo_radius=(2, 3))
    substr = ("Input ``halo_radius`` must be a float")
    assert substr in err.value.args[0]


def test_mc_generate_nfw_radial_positions4():

    with pytest.raises(HalotoolsError) as err:
        __ = mc_generate_nfw_radial_positions(num_pts=10, halo_radius=0.5, conc=(2, 3))
    substr = ("Input ``conc`` must be a float")
    assert substr in err.value.args[0]
