""" Module providing unit-testing for the functions in
`~halotools.mock_observables.catalog_analysis_helpers` module.
"""
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase

import numpy as np

from astropy.tests.helper import pytest
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
