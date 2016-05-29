#!/usr/bin/env python
import numpy as np
from astropy.table import Table
from copy import copy, deepcopy

from astropy.tests.helper import pytest
from unittest import TestCase
import warnings

from ..occupation_model_template import OccupationComponent

from ... import model_defaults
from ...factories import PrebuiltHodModelFactory, HodModelFactory

from ....sim_manager import FakeSim
from ....custom_exceptions import HalotoolsError

__all__ = ('TestOccupationComponent', )


class TestOccupationComponent(TestCase):

    def setUp(self):
        self.good_constructor_kwargs = ({
            'gal_type': 'centrals',
            'threshold': 11,
            'prim_haloprop_key': 'halo_mvir',
            'upper_occupation_bound': 1
            })

    def test_required_kwargs1(self):

        class MyOccupationComponent(OccupationComponent):

            def __init__(self):
                OccupationComponent.__init__(self)

        with pytest.raises(KeyError) as err:
            model = MyOccupationComponent()
        substr = 'gal_type'
        assert substr in err.value.args[0]

    def test_required_kwargs2(self):

        class MyOccupationComponent(OccupationComponent):

            def __init__(self):
                OccupationComponent.__init__(self, gal_type='satellites')

        with pytest.raises(KeyError) as err:
            model = MyOccupationComponent()
        substr = 'threshold'
        assert substr in err.value.args[0]

    def test_required_kwargs3(self):

        class MyOccupationComponent(OccupationComponent):

            def __init__(self):
                OccupationComponent.__init__(self,
                    gal_type='satellites', threshold=11,
                    prim_haloprop_key='halo_mvir')

        with pytest.raises(KeyError) as err:
            model = MyOccupationComponent()
        substr = 'upper_occupation_bound'
        assert substr in err.value.args[0]

    def test_required_methods1(self):

        constructor_kwargs = self.good_constructor_kwargs

        class MyOccupationComponent(OccupationComponent):

            def __init__(self):
                OccupationComponent.__init__(self, **constructor_kwargs)

        with pytest.raises(SyntaxError) as err:
            model = MyOccupationComponent()
        substr = 'implement a method named mean_occupation'
        assert substr in err.value.args[0]

    def test_required_methods2(self):

        constructor_kwargs = self.good_constructor_kwargs

        class MyOccupationComponent(OccupationComponent):

            def __init__(self):
                OccupationComponent.__init__(self, **constructor_kwargs)

            def mean_occupation(self, **kwargs):
                return None

        model = MyOccupationComponent()

    def test_nonstandard_upper_occupation_bound1(self):

        constructor_kwargs = self.good_constructor_kwargs
        constructor_kwargs['upper_occupation_bound'] = 2

        class MyOccupationComponent(OccupationComponent):

            def __init__(self):
                OccupationComponent.__init__(self, **constructor_kwargs)

            def mean_occupation(self, **kwargs):
                return None

        model = MyOccupationComponent()
        with pytest.raises(HalotoolsError) as err:
            _ = model.mc_occupation(prim_haloprop=1e10)
        substr = "write your own ``mc_occupation`` method that overrides the method "
        assert substr in err.value.args[0]

    @pytest.mark.slow
    def test_nonstandard_upper_occupation_bound2(self):

        zheng_model = PrebuiltHodModelFactory('zheng07')

        constructor_kwargs = self.good_constructor_kwargs
        constructor_kwargs['upper_occupation_bound'] = 2

        class MyOccupationComponent(OccupationComponent):

            def __init__(self, threshold):
                OccupationComponent.__init__(self, gal_type='centrals',
                    threshold=threshold, upper_occupation_bound=2)

            def mean_occupation(self, **kwargs):
                return None

            def mc_occupation(self, **kwargs):
                table = kwargs['table']
                result = np.random.randint(0, 2, len(table))
                table['halo_num_centrals'] = result
                return result

        occu_model = MyOccupationComponent(zheng_model.threshold)
        assert hasattr(occu_model, '_galprop_dtypes_to_allocate')
        dt = occu_model._galprop_dtypes_to_allocate
        assert 'halo_num_centrals' in dt.names

        new_model = HodModelFactory(baseline_model_instance=zheng_model,
            centrals_occupation=occu_model)
        halocat = FakeSim()
        new_model.populate_mock(halocat)
        cenmask = new_model.mock.galaxy_table['gal_type'] == 'centrals'
        assert len(new_model.mock.galaxy_table[cenmask]) > 0

    @pytest.mark.slow
    def test_nonstandard_upper_occupation_bound3(self):

        zheng_model = PrebuiltHodModelFactory('zheng07')

        class MySatelliteOccupation(OccupationComponent):

            def __init__(self, threshold):

                OccupationComponent.__init__(self,
                    gal_type='satellites',
                    threshold=threshold,
                    upper_occupation_bound=5)

            def mean_occupation(self, **kwargs):
                table = kwargs['table']
                return np.zeros(len(table)) + 2.5

            def mc_occupation(self, **kwargs):
                table = kwargs['table']
                meanocc = self.mean_occupation(**kwargs)
                result = np.where(meanocc < 2.5, 0, 5)
                table['halo_num_satellites'] = result
                return result

        occ = MySatelliteOccupation(zheng_model.threshold)
        new_model = HodModelFactory(baseline_model_instance=zheng_model,
            satellites_occupation=occ)
        halocat = FakeSim()
        new_model.populate_mock(halocat)

    @pytest.mark.slow
    def test_nonexistent_prim_haloprop_key(self):

        zheng_model = PrebuiltHodModelFactory('zheng07')

        constructor_kwargs = self.good_constructor_kwargs

        class MyOccupationComponent(OccupationComponent):

            def __init__(self, threshold):
                OccupationComponent.__init__(self, gal_type='centrals',
                    threshold=threshold, upper_occupation_bound=1)

            def mean_occupation(self, **kwargs):
                table = kwargs['table']
                return np.zeros(len(table)) + 0.1

        occu_model = MyOccupationComponent(zheng_model.threshold)
        new_model = HodModelFactory(baseline_model_instance=zheng_model,
            centrals_occupation=occu_model)
        assert not hasattr(new_model, 'prim_haloprop_key')

        halocat = FakeSim()
        new_model.populate_mock(halocat)

    def tearDown(self):
        pass
