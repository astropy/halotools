#!/usr/bin/env python

from unittest import TestCase
import pytest

import numpy as np 
from astropy.table import Table

from ..abhod_components import HeavisideCenAssemBiasModel
from ...sim_manager import FakeSim
from ..hod_components import Zheng07Cens, Leauthaud11Cens
from .. import model_defaults


class TestAssembias(TestCase):

    def setup_class(self):

        Npts = 1e4
        mass = np.zeros(Npts) + 1e12
        conc = np.linspace(0, 10, Npts)

        self.halos12 = Table({model_defaults.prim_haloprop_key: mass, model_defaults.sec_haloprop_key: conc})

def test_assembias_cens():

    std_cens = Zheng07Cens()
    assembias_model = HeavisideCenAssemBiasModel(standard_cen_model = std_cens)

