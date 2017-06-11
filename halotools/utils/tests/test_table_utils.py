"""
"""
import numpy as np
from unittest import TestCase
import pytest
from astropy.table import Table
from astropy.utils.misc import NumpyRNGContext

from ..table_utils import SampleSelector, compute_conditional_percentiles

from ...sim_manager import FakeSim

__all__ = ('test_split_sample1', 'TestComputeConditionalPercentiles')

fixed_seed = 43


def test_split_sample1():
    """ Verify that the split_sample function successfully divides a sample in half
    """
    t = Table({'x': np.arange(10)})

    percentiles = 0.5
    result = SampleSelector.split_sample(table=t, key='x', percentiles=percentiles)

    assert len(result) == 2
    assert set(result[0]['x']) == set((0, 1, 2, 3, 4))
    assert set(result[1]['x']) == set((5, 6, 7, 8, 9))
    assert len(result[0]) == 5
    assert len(result[1]) == 5


def test_split_sample2():
    """ Verify that split_sample raises the appropriate exception when
    attempting to split the sample too finely
    """
    t = Table({'x': np.arange(10)})

    with pytest.raises(ValueError) as err:
        SampleSelector.split_sample(table=t[0:4], key='x',
            percentiles=[0.1, 0.2, 0.3, 0.4, 0.5])
    substr = "Input length of percentiles must be less than input table length"
    assert substr in err.value.args[0]


def test_split_sample3():
    """ Verify that split_sample raises the appropriate exception when
    attempting to split the sample too finely
    """
    t = Table({'x': np.arange(10)})

    with pytest.raises(ValueError) as err:
        SampleSelector.split_sample(table=t[0:4], key='x',
            percentiles=[0.1, 0.1, 0.95])
    substr = "The input percentiles spacing is too fine."
    assert substr in err.value.args[0]


def test_split_sample4():
    """ Verify that split_sample raises the appropriate exception when
    trying to split the sample on a non-existent key
    """
    t = Table({'x': np.arange(10)})

    with pytest.raises(KeyError) as err:
        SampleSelector.split_sample(table=t, key='y',
            percentiles=0.5)
    substr = "The ``y`` key does not appear in the table you are trying"
    assert substr in err.value.args[0]


def test_split_sample5():
    """ Verify that split_sample raises the appropriate exception when
    passing in a structured Numpy array
    """
    t = Table({'x': np.arange(10)})

    with pytest.raises(TypeError) as err:
        SampleSelector.split_sample(table=np.array(t), key='x',
            percentiles=0.5)
    substr = "Input table must be an Astropy Table instance"
    assert substr in err.value.args[0]


class TestComputeConditionalPercentiles(TestCase):

    def setUp(self):
        Npts = int(1e4)
        mass1 = np.zeros(int(Npts/2)) + 1e12
        mass2 = np.zeros(int(Npts/2)) + 1e14
        mass = np.append(mass1, mass2)
        zform1 = np.linspace(0, 10, int(Npts/2))
        zform2 = np.linspace(20, 30, int(Npts/2))
        zform = np.append(zform1, zform2)

        d = {'halo_mvir': mass, 'halo_zform': zform}
        t = Table(d)
        with NumpyRNGContext(fixed_seed):
            randomizer = np.random.random(len(t))
        random_idx = np.argsort(randomizer)
        t = t[random_idx]
        self.custom_halo_table = t

        fakesim = FakeSim()
        self.fake_halo_table = fakesim.halo_table

    def test_custom_halo_table(self):

        assert self.custom_halo_table['halo_zform'].max() == 30
        assert self.custom_halo_table['halo_zform'].min() == 0

    def test_fake_halo_table(self):

        percentiles = compute_conditional_percentiles(
                table=self.fake_halo_table,
                prim_haloprop_key='halo_mvir',
                sec_haloprop_key='halo_vmax')
        split = percentiles < 0.5
        low_vmax, high_vmax = self.fake_halo_table[split], self.fake_halo_table[np.invert(split)]

    def test_custom_halo_table2(self):
        prim_haloprop_bin_boundaries = [1e10, 1e13, 1e15]

        manual_mass_split = self.custom_halo_table['halo_mvir'] < 1e13
        manual_low_mass = self.custom_halo_table[manual_mass_split]
        manual_high_mass = self.custom_halo_table[np.invert(manual_mass_split)]
        assert np.all(manual_low_mass['halo_mvir'] == 1e12)
        assert np.all(manual_high_mass['halo_mvir'] == 1e14)
        assert manual_low_mass['halo_zform'].max() == 10
        assert manual_low_mass['halo_zform'].min() == 0
        assert manual_high_mass['halo_zform'].max() == 30
        assert manual_high_mass['halo_zform'].min() == 20

        percentiles = compute_conditional_percentiles(
                table=self.custom_halo_table,
                prim_haloprop_key='halo_mvir',
                sec_haloprop_key='halo_zform',
                prim_haloprop_bin_boundaries=prim_haloprop_bin_boundaries)

        assert np.all(percentiles <= 1)
        assert np.all(percentiles >= 0)
        assert np.any(percentiles > 0.9)
        assert np.any(percentiles < 0.1)

        low_mass_percentiles = percentiles[manual_mass_split]
        assert np.all(low_mass_percentiles <= 1)
        assert np.all(low_mass_percentiles >= 0)
        assert np.any(low_mass_percentiles > 0.9)
        assert np.any(low_mass_percentiles < 0.1)

        high_mass_percentiles = percentiles[np.invert(manual_mass_split)]
        assert np.all(high_mass_percentiles <= 1)
        assert np.all(high_mass_percentiles >= 0)
        assert np.any(high_mass_percentiles > 0.9)
        assert np.any(high_mass_percentiles < 0.1)
        assert set(low_mass_percentiles) == set(high_mass_percentiles)

        low_mass_split = low_mass_percentiles < 0.5
        low_mass_low_zform = manual_low_mass[low_mass_split]
        low_mass_high_zform = manual_low_mass[np.invert(low_mass_split)]
        assert 0 <= low_mass_low_zform['halo_zform'].max() <= 5
        assert 5 <= low_mass_high_zform['halo_zform'].max() <= 10

        high_mass_split = high_mass_percentiles < 0.5
        high_mass_low_zform = manual_high_mass[high_mass_split]
        high_mass_high_zform = manual_high_mass[np.invert(high_mass_split)]
        assert 20 <= high_mass_low_zform['halo_zform'].max() <= 25
        assert 25 <= high_mass_high_zform['halo_zform'].max() <= 30

        split = percentiles <= 0.5
        low_zform, high_zform = self.custom_halo_table[split], self.custom_halo_table[np.invert(split)]
        assert len(low_zform) == len(high_zform)

    def tearDown(self):
        del self.fake_halo_table
        del self.custom_halo_table
