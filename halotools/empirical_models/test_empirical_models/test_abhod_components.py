#!/usr/bin/env python

from unittest import TestCase
import pytest
from copy import copy

import numpy as np 
from astropy.table import Table

from ..abhod_components import HeavisideCenAssemBiasModel
from ...sim_manager import FakeSim
from ..hod_components import Zheng07Cens, Leauthaud11Cens
from .. import model_defaults
from ...utils.table_utils import SampleSelector


class TestAssembias(TestCase):

    def setup_class(self):

        Npts = 1e4
        mass = np.zeros(Npts) + 1e12
        conc = np.linspace(0, 10, Npts)

        self.halos12 = Table({model_defaults.prim_haloprop_key: mass, model_defaults.sec_haloprop_key: conc})

    # def test_assembias_cens(self):

    #     std_cens = Zheng07Cens()
    #     assembias_model = HeavisideCenAssemBiasModel(standard_cen_model = std_cens)
    #     assembias_model.param_dict['frac_dNmax'] = 0.5

    #     abkey = assembias_model.sec_haloprop_percentile_key

    #     middle_index = len(self.halos12)/2
    #     self.halos12[abkey] = 0.
    #     self.halos12[abkey][middle_index:] = 1.
    #     assert np.mean(self.halos12[abkey]) == 0.5

    #     self.halos12['mean_ncen'] = assembias_model.mean_occupation(halo_table = self.halos12)
    #     hlo, hhi = SampleSelector.split_sample(table=self.halos12, key=abkey, percentiles=0.5)
    #     assert np.all(hlo[abkey] == 0)
    #     assert np.all(hhi[abkey] == 1)

    #     lomean = hlo['mean_ncen'].mean()
    #     himean = hhi['mean_ncen'].mean()

    #     np.testing.assert_array_almost_equal(hlo['mean_ncen'], lomean, decimal=3)
    #     np.testing.assert_array_almost_equal(hhi['mean_ncen'], himean, decimal=3)

    #     assert lomean > himean


















