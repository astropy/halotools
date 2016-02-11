#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
from copy import deepcopy 

from collections import Counter

import numpy as np 

from astropy.tests.helper import pytest
from astropy.table import Table 

from ..aggregation import add_new_table_column

from ...sim_manager import FakeSim

from ...custom_exceptions import HalotoolsError

__all__ = ['TestAggregation']

class TestAggregation(TestCase):
    """ Class providing tests of the `~halotools.utils.aggregation`. 
    """
    def setUp(self):
        fake_sim = FakeSim()
        self.halo_table = fake_sim.halo_table

    def test_argument_checks1(self):
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'abc'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = ['abc']
        sorting_keys = None

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(5, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "The input ``table`` must be an Astropy `~astropy.table.Table`"
        assert substr in err.value.message

    def test_argument_checks2(self):
        new_colname = 'halo_mvir'
        new_coltype = 'f4'
        grouping_key = 'abc'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = ['abc']
        sorting_keys = None

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "The input ``new_colname`` cannot be an existing column of the input ``table``"
        assert substr in err.value.message

    def test_argument_checks3(self):
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'abc'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = ['abc']
        sorting_keys = None

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "The input ``grouping_key`` must be an existing column of the input ``table``"
        print(err.value.message)
        assert substr in err.value.message
        
    def test_argument_checks4(self):
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = 5
        colnames_needed_by_function = ['abc']
        sorting_keys = None

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "The input ``aggregation_function`` must be a callable function"
        assert substr in err.value.message
        
    def test_argument_checks5(self):
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = 3
        sorting_keys = None

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "The input ``colnames_needed_by_function`` must be an iterable sequence"
        assert substr in err.value.message

    def test_argument_checks6(self):
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = 'abc'
        sorting_keys = None

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "Your input ``colnames_needed_by_function`` should be a"
        assert substr in err.value.message
        substr = "list of strings, not a single string"
        assert substr in err.value.message

    def test_argument_checks7(self):
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = ['abc']
        sorting_keys = None

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "Each element of the input ``colnames_needed_by_function`` must be"
        assert substr in err.value.message
        substr = "an existing column name of the input ``table``"
        assert substr in err.value.message

    def test_argument_checks8(self):
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = ['halo_hostid']
        sorting_keys = 5

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "The input ``sorting_keys`` must be an iterable sequence"
        assert substr in err.value.message

    def test_argument_checks9(self):
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = ['halo_hostid']
        sorting_keys = 'def'

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "Your input ``sorting_keys`` should be a"
        assert substr in err.value.message
        substr = "list of strings, not a single string"
        assert substr in err.value.message

    def test_argument_checks10(self):
        """ Verify that the aggregation function raises an exception when 
        an element of the input ``sorting_keys`` is not  
        an existing column name of the input ``table``
        """
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = ['halo_hostid']
        sorting_keys = ['def']

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "Each element of the input ``sorting_keys`` must be"
        assert substr in err.value.message
        substr = "an existing column name of the input ``table``"
        assert substr in err.value.message

    def test_argument_checks11(self):
        """ Verify that the aggregation function raises an exception when 
        the first element of the input ``sorting_keys`` does not equal 
        the input ``grouping_key``. 
        """
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = lambda x: 5
        colnames_needed_by_function = ['halo_hostid']
        sorting_keys = ['halo_mvir']

        with pytest.raises(HalotoolsError) as err:
            _ = add_new_table_column(self.halo_table, new_colname, new_coltype, 
                grouping_key, aggregation_function, 
                colnames_needed_by_function, 
                sorting_keys=sorting_keys)
        substr = "The first element of the input ``sorting_keys`` must be"
        assert substr in err.value.message
        substr = "equal to the input ``grouping_key``"
        assert substr in err.value.message

    def test_function_correctness1(self):
        """ Verify that the aggregation function can correctly compute 
        and broadcast the result of a trivial group-wise function. 
        """
        new_colname = 'abc'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = lambda x: x
        colnames_needed_by_function = ['halo_hostid']
        sorting_keys = ['halo_hostid']

        temp_table = deepcopy(self.halo_table)
        add_new_table_column(
            temp_table, new_colname, new_coltype, 
            grouping_key, aggregation_function, 
            colnames_needed_by_function, 
            sorting_keys=sorting_keys)
        assert new_colname in temp_table.keys()
        assert np.all(temp_table[new_colname] == temp_table[colnames_needed_by_function[0]])

        del temp_table

    def test_function_correctness2(self):
        """ Verify that the aggregation function can correctly compute 
        and broadcast the values of the group-wise host halo mass. 
        """
        new_colname = 'halo_mhost'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'
        aggregation_function = lambda x: x[0]
        colnames_needed_by_function = ['halo_mvir']
        sorting_keys = ['halo_hostid', 'halo_upid']

        temp_table = deepcopy(self.halo_table)
        add_new_table_column(
            temp_table, new_colname, new_coltype, 
            grouping_key, aggregation_function, 
            colnames_needed_by_function, 
            sorting_keys=sorting_keys)
        assert new_colname in temp_table.keys()

        host_mask = temp_table['halo_upid'] == -1
        assert np.allclose(temp_table['halo_mhost'][host_mask], 
            temp_table['halo_mvir'][host_mask])

        assert np.any(temp_table['halo_mhost'][~host_mask] != 
            temp_table['halo_mvir'][~host_mask])

        del temp_table

    def test_function_correctness3(self):
        """ Verify that the aggregation function can correctly compute 
        and broadcast the values of the group-wise mean mass-weighted spin. 
        """
        new_colname = 'mean_mass_weighted_spin'
        new_coltype = 'f4'
        grouping_key = 'halo_hostid'

        colnames_needed_by_function = ['halo_mvir', 'halo_spin']
        sorting_keys = ['halo_hostid']

        def mean_mass_weighted_spin(mass, spin):
            return sum(mass*spin)/float(len(mass))
        aggregation_function = mean_mass_weighted_spin

        temp_table = deepcopy(self.halo_table)
        add_new_table_column(
            temp_table, new_colname, new_coltype, 
            grouping_key, aggregation_function, 
            colnames_needed_by_function, 
            sorting_keys=sorting_keys)
        assert new_colname in temp_table.keys()

        data = Counter(temp_table[grouping_key])
        _ = data.most_common()
        groupids = np.array([elt[0] for elt in _])
        richnesses = np.array([elt[1] for elt in _])
        stride_length = int(len(groupids)/20.)

        for groupid, richness in zip(groupids[::stride_length], richnesses[::stride_length]):
            idx = np.where(temp_table[grouping_key] == groupid)[0]
            group = temp_table[idx]
            assert len(group) == richness
            correct_result = np.mean(group['halo_mvir']*group['halo_spin'])
            returned_result = group['mean_mass_weighted_spin'][0]
            np.testing.assert_approx_equal(correct_result, returned_result)
        
        del temp_table





