"""
"""
from __future__ import absolute_import, division, print_function

from unittest import TestCase
from collections import Counter
import numpy as np
import pytest

from ..group_member_generator import group_member_generator

from ...sim_manager import FakeSim

__all__ = ['TestGroupMemberGenerator']


class TestGroupMemberGenerator(TestCase):
    """ Class providing tests of the `~halotools.utils.aggregation`.
    """

    def setUp(self):
        fake_sim = FakeSim()
        self.halo_table = fake_sim.halo_table
        self.halo_table.sort('halo_hostid')

    def test_argument_checks1(self):
        """ Verify that an informative exception is raised when
        passing in data that is not a table or structured array.
        """
        grouping_key = 'halo_hostid'
        requested_columns = ['halo_mvir']

        with pytest.raises(TypeError) as err:
            g = group_member_generator(4,
                grouping_key, requested_columns)
            for _ in g:
                pass
        substr = "The input ``data`` must be an Astropy Table or Numpy Structured Array"
        assert substr in err.value.args[0]

    def test_argument_checks2(self):
        """ Verify that an informative exception is raised when
        passing in a non-iterable for ``requested_columns``.
        """
        grouping_key = 'halo_hostid'
        requested_columns = 5

        with pytest.raises(TypeError) as err:
            g = group_member_generator(self.halo_table,
                grouping_key, requested_columns)
            for _ in g:
                pass
        substr = "The input ``requested_columns`` must be an iterable sequence"
        assert substr in err.value.args[0]

    def test_argument_checks3(self):
        """ Verify that an informative exception is raised when
        passing in unacceptable ``requested_columns``.
        """
        grouping_key = 'halo_hostid'
        requested_columns = ['Jose Canseco']

        with pytest.raises(KeyError) as err:
            g = group_member_generator(self.halo_table,
                grouping_key, requested_columns)
            for _ in g:
                pass
        substr = "Each element of the input ``requested_columns`` must be"
        assert substr in err.value.args[0]

    def test_argument_checks4(self):
        """ Verify that an informative exception is raised when
        passing in unacceptable ``grouping_key``.
        """
        grouping_key = 'Jose Canseco'
        requested_columns = ['halo_hostid']

        with pytest.raises(KeyError) as err:
            g = group_member_generator(self.halo_table,
                grouping_key, requested_columns)
            for _ in g:
                pass
        substr = "Input ``grouping_key`` must be a column name of the input ``data``"
        assert substr in err.value.args[0]

    def test_argument_checks5(self):
        """ Verify that an informative exception is raised when
        passing in a single string for ``requested_columns``, rather than a list.
        """
        grouping_key = 'halo_hostid'
        requested_columns = 'Jose Canseco'

        with pytest.raises(KeyError) as err:
            g = group_member_generator(self.halo_table,
                grouping_key, requested_columns)
            for _ in g:
                pass
        substr = "list of strings, not a single string"
        assert substr in err.value.args[0]

    def test_argument_checks6(self):
        """ Verify that an informative exception is raised when
        the input data is not appropriately sorted.
        """
        grouping_key = 'halo_mvir'
        requested_columns = ['halo_mvir']

        with pytest.raises(ValueError) as err:
            g = group_member_generator(self.halo_table,
                grouping_key, requested_columns)
            for _ in g:
                pass
        substr = "Your input ``data`` must be sorted so that"
        assert substr in err.value.args[0]

    def test_function_correctness1(self):
        """ Verify that the aggregation function can correctly compute
        and broadcast the result of a trivial group-wise function.
        """
        grouping_key = 'halo_hostid'
        requested_columns = ['halo_hostid']
        gen = group_member_generator(self.halo_table,
            grouping_key, requested_columns)

        result = np.zeros(len(self.halo_table))
        for group_data in gen:
            first, last, group_array_list = group_data
            result[first:last] = group_array_list[0]
        assert np.all(result == self.halo_table['halo_hostid'])

    def test_function_correctness2(self):
        """ Verify that the aggregation function can correctly identify
        the first group member
        """
        self.halo_table.sort(keys=['halo_hostid', 'halo_upid'])

        grouping_key = 'halo_hostid'
        requested_columns = ['halo_upid']
        gen = group_member_generator(self.halo_table,
            grouping_key, requested_columns)

        result = np.zeros(len(self.halo_table))
        for group_data in gen:
            first, last, group_array_list = group_data
            result[first:last] = group_array_list[0][0]

        assert np.all(result == -1)

    def test_function_correctness3(self):
        """ Verify that the aggregation function can correctly compute
        and broadcast the values of the group-wise host halo mass.
        """
        self.halo_table.sort(keys=['halo_hostid', 'halo_upid'])

        grouping_key = 'halo_hostid'
        requested_columns = ['halo_mvir']
        gen = group_member_generator(self.halo_table,
            grouping_key, requested_columns)

        result = np.zeros(len(self.halo_table))
        for group_data in gen:
            first, last, group_array_list = group_data
            result[first:last] = group_array_list[0][0]

        host_mask = self.halo_table['halo_upid'] == -1
        assert np.all(self.halo_table['halo_mvir'][host_mask] == result[host_mask])
        assert np.any(self.halo_table['halo_mvir'][~host_mask] != result[~host_mask])

    def test_function_correctness4(self):
        """ Verify that the aggregation function can correctly compute
        and broadcast the values of the group-wise mean mass-weighted spin.
        """
        self.halo_table.sort(keys=['halo_hostid', 'halo_upid'])

        grouping_key = 'halo_hostid'
        requested_columns = ['halo_mvir', 'halo_spin']
        gen = group_member_generator(self.halo_table,
            grouping_key, requested_columns)

        result = np.zeros(len(self.halo_table))
        for group_data in gen:
            first, last, group_array_list = group_data
            mass = group_array_list[0]
            spin = group_array_list[1]
            result[first:last] = sum(mass*spin)/float(len(mass))
        self.halo_table['mean_mass_weighted_spin'] = result

        data = Counter(self.halo_table[grouping_key])
        _ = data.most_common()
        groupids = np.array([elt[0] for elt in _])
        richnesses = np.array([elt[1] for elt in _])
        stride_length = int(len(groupids)/20.)

        for groupid, richness in zip(groupids[::stride_length], richnesses[::stride_length]):
            idx = np.where(self.halo_table[grouping_key] == groupid)[0]
            group = self.halo_table[idx]
            assert len(group) == richness
            correct_result = np.mean(group['halo_mvir']*group['halo_spin'])
            returned_result = group['mean_mass_weighted_spin'][0]
            np.testing.assert_approx_equal(correct_result, returned_result)

    def tearDown(self):
        del self.halo_table
