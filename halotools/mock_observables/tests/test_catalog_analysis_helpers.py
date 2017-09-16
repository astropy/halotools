""" Module providing unit-testing for the functions in
`~halotools.mock_observables.catalog_analysis_helpers` module.
"""
from __future__ import absolute_import, division, print_function

from unittest import TestCase

import numpy as np

import pytest
from astropy.utils.misc import NumpyRNGContext

from .. import catalog_analysis_helpers as cat_helpers
from ..catalog_analysis_helpers import cuboid_subvolume_labels

from ...sim_manager import FakeSim

from .cf_helpers import generate_locus_of_3d_points

from ...custom_exceptions import HalotoolsError

__all__ = ('TestCatalogAnalysisHelpers', )

fixed_seed = 43


def test_cuboid_subvolume_labels_bounds_checking():
    Npts = 100
    with NumpyRNGContext(fixed_seed):
        good_sample = np.random.random((Npts, 3))
        bad_sample = np.random.random((Npts, 2))

    good_Nsub1 = 3
    good_Nsub2 = (4, 4, 4)
    bad_Nsub = (3, 3)

    good_Lbox = 1
    good_Lbox2 = (1, 1, 1)
    bad_Lbox = (3, 3)

    with pytest.raises(TypeError) as err:
        cuboid_subvolume_labels(bad_sample, good_Nsub1, good_Lbox)
    substr = "Input ``sample`` must have shape (Npts, 3)"
    assert substr in err.value.args[0]

    with pytest.raises(TypeError) as err:
        cuboid_subvolume_labels(good_sample, bad_Nsub, good_Lbox2)
    substr = "Input ``Nsub`` must be a scalar or length-3 sequence"
    assert substr in err.value.args[0]

    with pytest.raises(TypeError) as err:
        cuboid_subvolume_labels(good_sample, good_Nsub2, bad_Lbox)
    substr = "Input ``Lbox`` must be a scalar or length-3 sequence"
    assert substr in err.value.args[0]


def test_cuboid_subvolume_labels_correctness():
    Npts = 100
    Nsub = 2
    Lbox = 1

    sample = generate_locus_of_3d_points(Npts, xc=0.1, yc=0.1, zc=0.1, seed=fixed_seed)
    labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    assert np.all(labels == 1)

    sample = generate_locus_of_3d_points(Npts, xc=0.9, yc=0.9, zc=0.9, seed=fixed_seed)
    labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    assert np.all(labels == 8)

    sample = generate_locus_of_3d_points(Npts, xc=0.1, yc=0.1, zc=0.9, seed=fixed_seed)
    labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    assert np.all(labels == 2)

    sample = generate_locus_of_3d_points(Npts, xc=0.1, yc=0.9, zc=0.1, seed=fixed_seed)
    labels, N_sub_vol = cuboid_subvolume_labels(sample, Nsub, Lbox)
    assert np.all(labels == 3)


def test_sign_pbc1():
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = np.array([0, 2, 4, 5, 0])

    s = cat_helpers.sign_pbc(x1, x2)
    assert np.all(s == [1, 0, -1, -1, 1])

    s = cat_helpers.sign_pbc(x1, x2, equality_fill_val=9)
    assert np.all(s == [1, 9, -1, -1, 1])


def test_sign_pbc2():
    x1 = np.array((1, 4, 6, 9.))

    x2 = np.array((0, 3, 5, 8))
    s = cat_helpers.sign_pbc(x1, x2, period=10)
    assert np.all(s == [1, 1, 1., 1])


def test_sign_pbc3():
    x1 = np.array((1, 4, 6, 9.))
    x2 = np.array((9, 9.9, 0, 0))

    s = cat_helpers.sign_pbc(x1, x2, period=10)
    assert np.all(s == (1, 1, -1, -1))

    s = cat_helpers.sign_pbc(x1, x2)
    assert np.all(s == (-1, -1, 1, 1))


def test_sign_pbc_catches_out_of_bounds():
    x1 = np.array((1, 4, 6, 9.))

    x2 = np.array((2, 5, 7, 10))
    with pytest.raises(ValueError) as err:
        __ = cat_helpers.sign_pbc(x1, x2, period=10)
    substr = "If period is not None, all values of x and y must be between [0, period)"
    assert substr in err.value.args[0]

    s = cat_helpers.sign_pbc(x1, x2)
    assert np.all(s == (-1, -1, -1, -1))


def test_relative_positions_and_velocities_catches_out_of_bounds():
    x1 = np.array((1, 4, 6, 10.))
    x2 = x1
    with pytest.raises(ValueError) as err:
        __ = cat_helpers.relative_positions_and_velocities(x1, x2, period=10)
    substr = "If period is not None, all values of x and y must be between [0, period)"
    assert substr in err.value.args[0]


def test_relative_positions_and_velocities1():
    """ In this test, x1 > x2 and PBCs are irrelevant
    """
    period = 10
    x1 = np.array((1, 4, 6, 9.))
    x2 = x1 - 1
    result = cat_helpers.relative_positions_and_velocities(x1, x2, period=period)
    assert np.all(result == np.ones(len(x1)))
    result = cat_helpers.relative_positions_and_velocities(x1, x2)
    assert np.all(result == np.ones(len(x1)))


def test_relative_positions_and_velocities2():
    """ In this test, x1 > x2 and PBCs impact the results
    """
    period = 10
    x1 = np.array((1, 4, 6, 9.))
    x2 = np.mod(x1 - 1 + period, period)
    result = cat_helpers.relative_positions_and_velocities(x1, x2, period=period)
    assert np.all(result == np.ones(len(x1)))


def test_relative_positions_and_velocities3():
    """ In this test, x1 < x2 and PBCs are irrelevant
    """
    period = 100
    x1 = np.array((1, 4, 6, 9.))
    x2 = x1 + 1
    result = cat_helpers.relative_positions_and_velocities(x1, x2, period=period)
    assert np.all(result == -np.ones(len(x1)))
    result = cat_helpers.relative_positions_and_velocities(x1, x2)
    assert np.all(result == -np.ones(len(x1)))


def test_relative_positions_and_velocities4():
    """ In this test, x1 < x2 and PBCs impact the results
    """
    period = 10
    x1 = np.array((1, 4, 6, 9.))
    x2 = np.zeros(len(x1))
    result = cat_helpers.relative_positions_and_velocities(x1, x2, period=period)
    assert np.all(result == np.array((1, 4, -4, -1)))
    result = cat_helpers.relative_positions_and_velocities(x1, x2)
    assert np.all(result == x1)


def test_relative_velocities1():
    """ In this test, x1 > x2 and PBCs are irrelevant
    """
    period = 100
    x1 = np.array((9, 9, 9, 9))
    x2 = x1 - 1
    v1 = np.array((100, 100, 100, 100))
    v2 = np.array((-100, -10, 90, 110))
    xrel, vrel = cat_helpers.relative_positions_and_velocities(x1, x2, period=period, v1=v1, v2=v2)
    assert np.all(vrel == (200, 110, 10, -10))
    xrel, vrel = cat_helpers.relative_positions_and_velocities(x1, x2, v1=v1, v2=v2)
    assert np.all(vrel == (200, 110, 10, -10))


def test_relative_velocities2():
    """ In this test, x1 > x2 and PBCs impact the results
    """
    period = 10
    x1 = np.array((9, 9, 9, 9))
    x2 = np.zeros(len(x1))
    v1 = np.array((100, 100, 100, 100))
    v2 = np.array((-100, -10, 90, 110))
    correct_result = np.array((-200, -110, -10, 10))
    xrel, vrel = cat_helpers.relative_positions_and_velocities(x1, x2, period=period, v1=v1, v2=v2)
    assert np.all(vrel == correct_result)
    xrel, vrel = cat_helpers.relative_positions_and_velocities(x1, x2, v1=v1, v2=v2)
    assert np.all(vrel == -correct_result)


def test_relative_velocities3():
    """ In this test, x1 < x2 and PBCs are irrelevant
    """
    period = 100
    x1 = np.array((9, 9, 9, 9))
    x2 = x1 + 1
    v1 = np.array((100, 100, 100, 100))
    v2 = np.array((-100, -10, 90, 110))
    xrel, vrel = cat_helpers.relative_positions_and_velocities(x1, x2, period=period, v1=v1, v2=v2)
    correct_result = np.array((-200, -110, -10, 10))
    assert np.all(vrel == correct_result)
    xrel, vrel = cat_helpers.relative_positions_and_velocities(x1, x2, v1=v1, v2=v2)
    assert np.all(vrel == correct_result)


def test_relative_velocities4():
    """ In this test, x1 < x2 and PBCs are irrelevant
    """
    period = 10
    x1 = np.zeros(4)
    x2 = np.zeros(4) + 9.
    v1 = np.array((100, 100, 100, 100))
    v2 = np.array((-100, -10, 90, 110))
    xrel, vrel = cat_helpers.relative_positions_and_velocities(x1, x2, period=period, v1=v1, v2=v2)
    correct_result = np.array((200, 110, 10, -10))
    assert np.all(vrel == correct_result)
    xrel, vrel = cat_helpers.relative_positions_and_velocities(x1, x2, v1=v1, v2=v2)
    assert np.all(vrel == -correct_result)


def test_return_xyz_formatted_array1():
    npts = 10
    period = [1, 2, 3]
    x = np.linspace(0.001, period[0]-0.001, npts)
    y = np.linspace(0.001, period[1]-0.001, npts)
    z = np.linspace(0.001, period[2]-0.001, npts)
    v = np.zeros(npts)
    result1 = cat_helpers.return_xyz_formatted_array(x, y, z)
    result2 = cat_helpers.return_xyz_formatted_array(x, y, z, velocity=v, velocity_distortion_dimension='x')
    result3 = cat_helpers.return_xyz_formatted_array(x, y, z, velocity=v, velocity_distortion_dimension='y')
    result4 = cat_helpers.return_xyz_formatted_array(x, y, z, velocity=v, velocity_distortion_dimension='z')
    result5 = cat_helpers.return_xyz_formatted_array(x, y, z, velocity=v,
        velocity_distortion_dimension='x', period=period)
    result6 = cat_helpers.return_xyz_formatted_array(x, y, z, velocity=v,
        velocity_distortion_dimension='y', period=period)

    assert np.all(result1 == result2)
    assert np.all(result1 == result3)
    assert np.all(result1 == result4)
    assert np.all(result1 == result5)
    assert np.all(result1 == result6)


def test_return_xyz_formatted_array2():
    """ verify that redshift keyword is operative
    """
    npts = int(1e4)
    x = np.linspace(0.001, 0.999, npts)
    y = np.linspace(0.001, 0.999, npts)
    z = np.linspace(0.001, 0.999, npts)
    v = np.random.normal(loc=0, scale=150, size=npts)
    result_z0 = cat_helpers.return_xyz_formatted_array(x, y, z,
        velocity=v, velocity_distortion_dimension='x', redshift=0)
    result_z1 = cat_helpers.return_xyz_formatted_array(x, y, z,
        velocity=v, velocity_distortion_dimension='x', redshift=1)
    assert not np.all(result_z0 == result_z1)


def test_return_xyz_formatted_array3():
    """ verify that cosmology keyword is operative
    """
    npts = int(1e4)
    x = np.linspace(0.001, 0.999, npts)
    y = np.linspace(0.001, 0.999, npts)
    z = np.linspace(0.001, 0.999, npts)
    v = np.random.normal(loc=0, scale=150, size=npts)

    from astropy.cosmology import WMAP5, Planck15
    result_z0a = cat_helpers.return_xyz_formatted_array(x, y, z,
        velocity=v, velocity_distortion_dimension='x', redshift=0.5, cosmology=WMAP5)
    result_z0b = cat_helpers.return_xyz_formatted_array(x, y, z,
        velocity=v, velocity_distortion_dimension='x', redshift=0.5, cosmology=Planck15)
    assert not np.all(result_z0a == result_z0b)


def test_return_xyz_formatted_array4():
    """ Verify consistent behavior between
    the `~halotools.mock_observables.return_xyz_formatted_array` function and the
    independently-written `~halotools.mock_observables.return_xyz_formatted_array` function.
    """
    npts = int(1e4)
    x = np.linspace(0.001, 0.999, npts)
    y = np.linspace(0.001, 0.999, npts)
    z = np.linspace(0.001, 0.999, npts)
    v = np.random.normal(loc=0, scale=0.5, size=npts)

    from astropy.cosmology import WMAP5
    result1 = cat_helpers.return_xyz_formatted_array(x, y, z,
        velocity=v, velocity_distortion_dimension='x', redshift=0.5, cosmology=WMAP5, period=1)

    result2 = cat_helpers.apply_zspace_distortion(x, v, 0.5, WMAP5, Lbox=1)
    assert np.allclose(result1[:, 0], result2)


class TestCatalogAnalysisHelpers(TestCase):
    """ Class providing tests of the `~halotools.mock_observables.catalog_analysis_helpers`.
    """

    def setUp(self):

        halocat = FakeSim()
        self.halo_table = halocat.halo_table
        self.Lbox = halocat.Lbox

    def test_mean_y_vs_x1(self):
        abscissa, mean, err = cat_helpers.mean_y_vs_x(
            self.halo_table['halo_mvir'], self.halo_table['halo_spin'])

    def test_mean_y_vs_x2(self):
        abscissa, mean, err = cat_helpers.mean_y_vs_x(
            self.halo_table['halo_mvir'], self.halo_table['halo_spin'],
            error_estimator='variance')

    def test_mean_y_vs_x3(self):
        with pytest.raises(HalotoolsError) as err:
            abscissa, mean, err = cat_helpers.mean_y_vs_x(
                self.halo_table['halo_mvir'], self.halo_table['halo_spin'],
                error_estimator='Jose Canseco')
        substr = "Input ``error_estimator`` must be either"
        assert substr in err.value.args[0]

    def test_return_xyz_formatted_array1(self):
        x, y, z = (self.halo_table['halo_x'],
            self.halo_table['halo_y'], self.halo_table['halo_z'])
        pos = cat_helpers.return_xyz_formatted_array(x, y, z)
        assert np.shape(pos) == (len(x), 3)

        mask = self.halo_table['halo_mvir'] >= 10**13.5
        masked_pos = cat_helpers.return_xyz_formatted_array(x, y, z, mask=mask)
        npts = len(self.halo_table[mask])
        assert np.shape(masked_pos) == (npts, 3)

        assert masked_pos.shape[0] < pos.shape[0]

        pos_zdist = cat_helpers.return_xyz_formatted_array(
            x, y, z, velocity=self.halo_table['halo_vz'],
            velocity_distortion_dimension='z')
        assert np.all(pos_zdist[:, 0] == pos[:, 0])
        assert np.all(pos_zdist[:, 1] == pos[:, 1])
        assert np.any(pos_zdist[:, 2] != pos[:, 2])
        assert np.all(abs(pos_zdist[:, 2] - pos[:, 2]) < 50)

        pos_zdist_pbc = cat_helpers.return_xyz_formatted_array(
            x, y, z, velocity=self.halo_table['halo_vz'],
            velocity_distortion_dimension='z',
            period=self.Lbox)
        assert np.all(pos_zdist_pbc[:, 0] == pos[:, 0])
        assert np.all(pos_zdist_pbc[:, 1] == pos[:, 1])
        assert np.any(pos_zdist_pbc[:, 2] != pos[:, 2])

        assert np.any(abs(pos_zdist_pbc[:, 2] - pos[:, 2]) > 50)

    def tearDown(self):
        del self.halo_table
