#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function, 
    unicode_literals)
__all__ = ['test_ConcMass']

import numpy as np
from astropy import cosmology
from astropy.table import Table

from ..conc_mass_models import ConcMass

from .... import model_defaults

from .....sim_manager import FakeSim, FakeMock

def test_ConcMass():
    """ Test the `~halotools.empirical_models.ConcMass` module. 
    Summary of tests is as follows: 
    
        * Returned concentrations satisfy :math:`0 < c < 100` for the full range of reasonable masses

        * Returns identical results regardless of argument choice

        * The :math:`c(M)` relation is monotonic over the full range of reasonable masses

    """
    dutton_maccio14_model = ConcMass(redshift=0, conc_mass_model = 'dutton_maccio14')

    Npts = 1e3
    mass = np.logspace(10, 15, Npts)
    conc = dutton_maccio14_model.compute_concentration(prim_haloprop=mass)
    assert np.all(conc > 1)
    assert np.all(conc < 100)
    assert np.all(np.diff(conc) < 0)

    conc = np.random.uniform(0, 100, Npts)
    t = Table({'conc': conc, 'halo_mvir': mass})
    direct_model = ConcMass(redshift=0, 
        conc_mass_model = 'direct_from_halo_catalog', 
        concentration_key = 'conc')
    conc_result = direct_model.compute_concentration(halo_table = t)

    assert np.all(conc_result >= model_defaults.min_permitted_conc)
    assert np.all(conc_result <= model_defaults.max_permitted_conc)










