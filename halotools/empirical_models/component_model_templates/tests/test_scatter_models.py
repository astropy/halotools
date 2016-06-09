""" Module providing unit-testing for the functions in
the `~halotools.empirical_models.component_model_templates.scatter_models` module
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.tests.helper import pytest

from ..scatter_models import LogNormalScatterModel

__all__=['test_nonzero_scatter', 'test_zero_scatter']

from halotools.sim_manager import FakeSim
halocat = FakeSim()
halo_table = halocat.halo_table

def test_nonzero_scatter():
    
    scatter_model = LogNormalScatterModel(scatter_abscissa=[10**10,10**12], scatter_ordinates=[0.1,0.1])
    
    scatter = scatter_model.scatter_realization(table = halo_table)
    
    assert len(scatter)==len(halo_table)


def test_zero_scatter():
    
    scatter_model = LogNormalScatterModel(scatter_abscissa=[10**10,10**12], scatter_ordinates=[0.0,0.0])
    
    scatter = scatter_model.scatter_realization(table = halo_table)
    
    assert len(scatter)==len(halo_table)
    assert np.all(scatter==0.0)

