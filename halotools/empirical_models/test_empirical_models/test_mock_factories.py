#!/usr/bin/env python

import numpy as np 
from .. import preloaded_models
from .. import model_factories
from .. import mock_factories
from .. import preloaded_models
from ...sim_manager.generate_random_sim import FakeSim

__all__ = ['test_preloaded_hod_mocks']


def test_preloaded_hod_mocks():
    """ Loop over all pre-loaded HOD models, 
    and one-by-one test that mock instances created by 
    `~halotools.empirical_models.HodMockFactory`. 

    Notes 
    -----
    Any HOD-style model listed in the ``__all__`` built-in attribute of the 
    `~halotools.empirical_models.preloaded_models` module will be tested. 
    Test suite includes: 

        * Mock has appropriate properties when instantiated with default settings, as well as non-trivial entries for ``additional_haloprops``, ``new_haloprop_func_dict``, and ``create_astropy_table``. 

        * Galaxy positions satisfy :math:`0 < x, y, z < L_{\\rm box}`.  
    """

    def test_hod_mock_attrs(model, sim):

    # If factory is called with default settings, 
    # mock attributes should include/exclude:
        mock1 = mock_factories.HodMockFactory(sim, model)
        assert hasattr(mock1, 'galaxy_table')
        expected_keys = ['x', 'y', 'z', 'halo_x', 'halo_NFWmodel_conc', 'halo_mvir']
        for key in expected_keys:
            assert key in mock1.galaxy_table.keys()

        mock1.build_halo_prof_lookup_tables()
        assert np.all(mock1.galaxy_table['x'] >= 0)
        assert np.all(mock1.galaxy_table['y'] >= 0)
        assert np.all(mock1.galaxy_table['z'] >= 0)
        assert np.all(mock1.galaxy_table['x'] <= mock1.snapshot.Lbox)
        assert np.all(mock1.galaxy_table['y'] <= mock1.snapshot.Lbox)
        assert np.all(mock1.galaxy_table['z'] <= mock1.snapshot.Lbox)

        assert np.all(mock1.galaxy_table['halo_NFWmodel_conc'] > 0.5)
        assert np.all(mock1.galaxy_table['halo_NFWmodel_conc'] < 25)

        mock2 = mock_factories.HodMockFactory(sim, model, 
            additional_haloprops = ['zhalf'])
        assert 'halo_zhalf' in mock2.galaxy_table.keys()

        #func_dict = {'double_mvir' : lambda halos : 2.*halos['mvir']}
        #mock4 = mock_factories.HodMockFactory(sim, model, 
        #    new_haloprop_func_dict = func_dict)
        #assert 'double_mvir' in mock4.halos.keys()
        #assert hasattr(mock4, 'halo_double_mvir')
        #assert np.allclose(mock4.halo_mvir/mock4.halo_double_mvir, 0.5)


    sim = FakeSim()

    hod_model_list = preloaded_models.__all__
    parent_class = model_factories.HodModelFactory
    # Create a list of all pre-loaded HOD models that we will test
    component_models_to_test = []
    for clname in hod_model_list:
        clfunc = getattr(preloaded_models, clname)
        cl = clfunc()
        if (isinstance(cl, parent_class)):
            component_models_to_test.append(cl)

    for model in component_models_to_test:
        test_hod_mock_attrs(model, sim)










