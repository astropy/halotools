#!/usr/bin/env python

import numpy as np 
from .. import preloaded_models
from .. import hod_factory
from .. import mock_factory
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
        expected_attr_list = ['Ngals', 'gal_NFWmodel_conc','halo_NFWmodel_conc','prim_haloprop_key','galaxy_table']
        excluded_attr_list = ['halo_conc']
        mock1 = mock_factory.HodMockFactory(sim, model)
        for attr in expected_attr_list:
            assert hasattr(mock1, attr)
        for attr in excluded_attr_list:
            assert hasattr(mock1, attr) is False
        assert mock1.create_astropy_table == True

        mock1.build_halo_prof_lookup_tables()
        mock1.bundle_into_table()
        assert hasattr(mock1, 'galaxy_table')
        assert np.all(mock1.pos > 0)
        assert np.all(mock1.pos < mock1.snapshot.Lbox)
        assert np.all(mock1.halo_NFWmodel_conc > 0.5)
        assert np.all(mock1.halo_NFWmodel_conc < 25.0)

        mock2 = mock_factory.HodMockFactory(sim, model, 
            additional_haloprops=['conc'])
        correct_haloprop_list = mock1.additional_haloprops
        correct_haloprop_list.append('conc')
        assert set(mock2.additional_haloprops) == set(correct_haloprop_list)
        expected_attr_list.append('halo_conc')
        for attr in expected_attr_list:
            assert hasattr(mock2, attr)

        mock3 = mock_factory.HodMockFactory(sim, model, 
            create_astropy_table=True)
        assert hasattr(mock3, 'galaxy_table')
        mock3 = mock_factory.HodMockFactory(sim, model, 
            create_astropy_table=False)
        assert hasattr(mock3, 'galaxy_table') is False
        mock3.bundle_into_table()
        assert hasattr(mock3, 'galaxy_table')

        func_dict = {'double_mvir' : lambda halos : 2.*halos['mvir']}
        mock4 = mock_factory.HodMockFactory(sim, model, 
            new_haloprop_func_dict = func_dict)
        assert 'double_mvir' in mock4.halos.keys()
        assert hasattr(mock4, 'halo_double_mvir')
        assert np.allclose(mock4.halo_mvir/mock4.halo_double_mvir, 0.5)


    sim = FakeSim()

    hod_model_list = preloaded_models.__all__
    parent_class = hod_factory.HodModelFactory
    # Create a list of all pre-loaded HOD models that we will test
    component_models_to_test = []
    for clname in hod_model_list:
        clfunc = getattr(preloaded_models, clname)
        cl = clfunc()
        if (isinstance(cl, parent_class)):
            component_models_to_test.append(cl)

    for model in component_models_to_test:
        test_hod_mock_attrs(model, sim)










