"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from astropy.table import Table
from astropy.utils.misc import NumpyRNGContext

from ...nfw_phase_space import NFWPhaseSpace
from ...nfw_profile import NFWProfile

from ....... import model_defaults


# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
from astropy.config.paths import _find_home
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False


__all__ = ('test_direct_from_halo_catalog1', 'test_direct_from_halo_catalog2',
        'test_direct_from_halo_catalog3')

fixed_seed = 43


classes_to_test = (NFWProfile, NFWPhaseSpace)
constructor_kwargs = ({'conc_mass_model': 'direct_from_halo_catalog',
            'concentration_key': 'conc'})


def test_direct_from_halo_catalog1():
    r""" Verify that the ``direct_from_halo_catalog`` conc-mass option
    behaves correctly with an input table with known concentrations.
    """
    for Model in classes_to_test:
        nfw = Model(**constructor_kwargs)

        Npts = int(10)
        mass = np.logspace(10, 15, Npts)
        cmin, cmax = model_defaults.min_permitted_conc, model_defaults.max_permitted_conc
        with NumpyRNGContext(fixed_seed):
            conc = np.random.uniform(cmin, cmax, Npts)
        t = Table({'conc': conc, 'halo_mvir': mass})
        conc_result = nfw.conc_NFWmodel(table=t)
        assert np.allclose(conc, conc_result)


def test_direct_from_halo_catalog2():
    r""" Verify that the ``direct_from_halo_catalog`` conc-mass option
    behaves correctly with an input table with known concentrations.

    This test verifies that concentrations below model_defaults.min_permitted_conc
    are correctly clipped.
    """
    for Model in classes_to_test:
        nfw = Model(**constructor_kwargs)

        Npts = int(10)
        mass = np.logspace(10, 15, Npts)
        cmin, cmax = model_defaults.min_permitted_conc-3, model_defaults.max_permitted_conc
        conc = np.linspace(cmin, cmax, Npts)
        t = Table({'conc': conc, 'halo_mvir': mass})
        conc_result = nfw.conc_NFWmodel(table=t)
        mask = conc >= model_defaults.min_permitted_conc
        assert np.allclose(conc[mask], conc_result[mask])
        assert np.all(conc_result[~mask] == model_defaults.min_permitted_conc)


def test_direct_from_halo_catalog3():
    r""" Verify that the ``direct_from_halo_catalog`` conc-mass option
    behaves correctly with an input table with known concentrations.

    This test verifies that concentrations above model_defaults.max_permitted_conc
    are correctly clipped.
    """
    for Model in classes_to_test:
        nfw = Model(**constructor_kwargs)

        Npts = int(10)
        mass = np.logspace(10, 15, Npts)
        cmin, cmax = model_defaults.min_permitted_conc, model_defaults.max_permitted_conc+3
        conc = np.linspace(cmin, cmax, Npts)
        t = Table({'conc': conc, 'halo_mvir': mass})
        conc_result = nfw.conc_NFWmodel(table=t)
        mask = conc <= model_defaults.max_permitted_conc
        assert np.allclose(conc[mask], conc_result[mask])
        assert np.all(conc_result[~mask] == model_defaults.max_permitted_conc)
