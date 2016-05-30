""" Module providing unit-testing for `~halotools.empirical_models.ConcMass` class.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.table import Table
from unittest import TestCase

from ..conc_mass_models import ConcMass

from .... import model_defaults

__all__ = ['TestConcMass']


class TestConcMass(TestCase):
    """ Tests of `~halotools.empirical_models.ConcMass` class.

    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests.
        """
        self.dutton_maccio14_model = ConcMass(redshift=0, conc_mass_model='dutton_maccio14')

        self.direct_model = ConcMass(redshift=0,
            conc_mass_model='direct_from_halo_catalog',
            concentration_key='conc')

    def test_dutton_maccio14(self):
        """ Tests of `~halotools.empirical_models.ConcMass.compute_concentration` method for the analytical *dutton_maccio14* option.
        Summary of tests is as follows:

            * Returned concentrations satisfy :math:`0 < c < 100` for the full range of reasonable masses

            * Returns identical results regardless of argument choice

            * The :math:`c(M)` relation is monotonic over the full range of reasonable masses

        """

        Npts = int(1e3)
        mass = np.logspace(10, 15, Npts)
        conc = self.dutton_maccio14_model.compute_concentration(prim_haloprop=mass)
        assert np.all(conc > 1)
        assert np.all(conc < 100)
        assert np.all(np.diff(conc) < 0)

    def test_direct_from_halo_catalog(self):
        """ Tests of `~halotools.empirical_models.ConcMass.compute_concentration` method for the *direct_from_halo_catalog* option.

        Require that the following are true:

        * Method behaves as the identity function whenever the halo concentrations are within the min/max range set in the `~halotools.empirical_models.model_defaults` module.

        * Method returns the min/max boundary values when passed halo concentrations that are out of bounds.
        """
        Npts = int(1e3)
        mass = np.logspace(10, 15, Npts)
        conc = np.random.uniform(0, 100, Npts)
        t = Table({'conc': conc, 'halo_mvir': mass})
        conc_result = self.direct_model.compute_concentration(table=t)

        within_bounds_mask = (conc >= model_defaults.min_permitted_conc) & (conc <= model_defaults.max_permitted_conc)
        assert np.all(conc_result[within_bounds_mask] == t['conc'][within_bounds_mask])

        out_of_bounds_mask = ~within_bounds_mask
        assert np.all(conc_result[out_of_bounds_mask] != t['conc'][out_of_bounds_mask])

        below_cmin_mask = conc < model_defaults.min_permitted_conc
        assert np.all(conc_result[below_cmin_mask] == model_defaults.min_permitted_conc)

        above_cmax_mask = conc > model_defaults.max_permitted_conc
        assert np.all(conc_result[above_cmax_mask] == model_defaults.max_permitted_conc)
