#!/usr/bin/env python

from unittest import TestCase
import pytest
from copy import copy

import numpy as np 
from astropy.table import Table

from ..assembias import HeavisideAssembias, AssembiasZheng07Cens

from .. import model_defaults
from ..hod_components import Zheng07Cens, Leauthaud11Cens
from ..sfr_components import BinaryGalpropInterpolModel
from ...sim_manager import FakeSim
from ...utils.table_utils import SampleSelector, compute_conditional_percentiles
from ...utils.array_utils import array_like_length as custom_len

class TestAssembias(TestCase):
    """
    """

    def setup_class(self):
        """
        """
        Npts = 1e4
        mass = np.zeros(Npts) + 1e12
        zform = np.linspace(0, 10, Npts)

        d1 = {'halo_mvir': mass, 'halo_zform': zform}
        self.toy_halo_table1 = Table(d1)

        halo_zform_percentile = (np.arange(Npts)+1) / float(Npts)
        halo_zform_percentile = 1. - halo_zform_percentile[::-1]
        d2 = {'halo_mvir': mass, 'halo_zform': zform, 'halo_zform_percentile': halo_zform_percentile}
        self.toy_halo_table2 = Table(d2)

        fakesim = FakeSim()
        self.fake_halo_table = fakesim.halo_table

    def test_assembias_zheng07_cens(self):
        abz = AssembiasZheng07Cens(sec_haloprop_key = 'halo_zform')

        baseline_result = abz.baseline_mean_occupation(halo_table = self.toy_halo_table2)
        np.testing.assert_allclose(baseline_result, 0.456686, rtol=1e-3)
        result = abz.mean_occupation(halo_table = self.toy_halo_table2)
        assert result.mean() == baseline_result.mean()

        mask = self.toy_halo_table2['halo_zform_percentile'] >= 0.5
        assert result[mask].mean() != result[np.invert(mask)].mean()

        oldmean = result[mask].mean()
        youngmean = result[np.invert(mask)].mean()
        derived_mean = 0.5*oldmean + 0.5*youngmean
        baseline_mean = baseline_result.mean()
        np.testing.assert_allclose(baseline_mean, derived_mean, rtol=1e-3)

















