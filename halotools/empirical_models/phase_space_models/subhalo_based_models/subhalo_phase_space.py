"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np

from .subhalo_selection_kernel import calculate_satellite_selection_mask

from ....utils import array_is_monotonic, crossmatch
from ....custom_exceptions import HalotoolsError

__author__ = ('Andrew Hearin', )
__all__ = ('SubhaloPhaseSpace', )


sorting_error_msg = ("\nIn order for the SubhaloPhaseSpace class to behave properly, \n"
    "the chosen ``subhalo_table_sorting_keys`` list should result in a subhalo_table with a \n"
    "``halo_hostid`` column that is monotonically increasing\n"
    "(although repeated entries in the ``halo_hostid`` column are permissible)\n")

default_inherited_subhalo_props_dict = (
    {'halo_id': ('halo_id', 'i8'),
    'halo_x': ('x', 'f8'),
    'halo_y': ('y', 'f8'),
    'halo_z': ('z', 'f8'),
    'halo_vx': ('vx', 'f8'),
    'halo_vy': ('vy', 'f8'),
    'halo_vz': ('vz', 'f8'),
    'halo_mpeak': ('halo_mpeak', 'f8')})


class SubhaloPhaseSpace(object):
    """ Class using subhalo information to model the phase space of satellite galaxies.

    For a demonstration of typical usage, see the :ref:`subhalo_phase_space_model_tutorial`.
    """

    def __init__(self, gal_type, host_haloprop_bins,
            inherited_subhalo_props_dict=default_inherited_subhalo_props_dict,
            binning_key='halo_mvir_host_halo', intra_halo_sorting_key='halo_mpeak',
            reverse_intra_halo_order=True, **kwargs):
        """
        Parameters
        ----------
        gal_type : string
            Name of the modeled galaxy population, e.g, 'satellites'.

        host_haloprop_bins : array_like
            Array used to bin subhalos into groups of a similar property,
            e.g., host halo mass. The ``host_haloprop_bins`` argument defines the bin edges.
            The user must also provide the column name storing the property
            that should be binned. This column name should be provided with the
            ``binning_key`` argument.

            For example, if you were binning subhalos by host halo mass,
            the ``binning_key`` argument would be ``halo_mvir_host_halo``,
            and ``host_haloprop_bins`` could be np.logspace(10, 15, 15).

            The purpose of ``host_haloprop_bins`` is to address the
            possibility that the desired number of satellites in a halo may
            exceed the number of subhalos in the halo. In such a case,
            randomly selected subhalos from the same bin are chosen.

        binning_key : string, optional
            Column name defining how the input subhalos will be grouped together
            by some host halo property such as virial mass.
            Default is ``halo_mvir_host_halo``.

        intra_halo_sorting_key : string, optional
            Column name defining how the subhalos will be sorted within each host halo.
            Subhalos appearing first are preferentially selected as satellites.
            Default is ``halo_mpeak``.
            This argument should be used together with reverse_intra_halo_order.

        reverse_intra_halo_order : bool, optional
            If set to True, subhalos with larger values of the intra_halo_sorting_key
            will be the first in each halo to be assigned galaxies.
            If False, preferential assignment goes to subhalos with smaller values.
            Default is True, for compatibility with the more typical properties
            such as ``halo_mpeak`` and ``halo_vpeak``.

        inherited_subhalo_props_dict : dict, optional
            Python dictionary determining which properties
            the modeled galaxies will inherit from their corresponding subhalo.
            Each key of the ``inherited_subhalo_props_dict`` dictionary
            gives the name of a column in the ``subhalo_table``
            that you wish to inherit.
            The value bound to each key is a tuple of two strings.
            The first string specifies the name you would like to give
            the inherited property in the ``galaxy_table``.
            The second string specifies the data type of the column, e.g., 'f4' or 'i8'.
            For example, suppose you wish to treat the x-position of the subhalo
            as the x-position of your satellite galaxy. Then you would have
            a ``halo_x`` key in the ``inherited_subhalo_props_dict`` dictionary,
            and the two-element tuple bound to it would be ('x', 'f4').

            Default value is set by the ``default_inherited_subhalo_props_dict``
            variable in the `~halotools.empirical_models.model_defaults` module.

        """
        self._mock_generation_calling_sequence = ['inherit_subhalo_properties']
        self._methods_to_inherit = ['inherit_subhalo_properties', 'preprocess_subhalo_table']

        self.list_of_haloprops_needed = list((intra_halo_sorting_key,
            binning_key, binning_key+'_bin_number', '_subhalo_inheritance_id', 'halo_hostid'))

        self.gal_type = gal_type
        self.host_haloprop_bins = host_haloprop_bins
        self.inherited_subhalo_props_dict = inherited_subhalo_props_dict
        self.binning_key = binning_key
        self.intra_halo_sorting_key = intra_halo_sorting_key
        self.reverse_intra_halo_order = reverse_intra_halo_order
        self.subhalo_table_sorting_keys = list((self.binning_key,
            '_subhalo_inheritance_id', self.intra_halo_sorting_key))

        self._additional_kwargs_dict = dict(
            inherit_subhalo_properties=['halo_table', 'subhalo_table', '_occupation', 'Lbox'])

        dt_arg = [(str(a[0]), str(a[1])) for a in inherited_subhalo_props_dict.values()]
        dt_arg.append((str('real_subhalo'), bool))
        self._galprop_dtypes_to_allocate = np.dtype(dt_arg)

        self.param_dict = {}

    def _retrieve_satellite_selection_idx(self, host_halo_table, subhalo_table, occupations,
            seed=None):
        """
        """
        try:
            assert len(occupations) == len(host_halo_table)
        except AssertionError:
            msg = ("The input `occupations`` array and ``host_halo_table`` \n"
                "must have equal length. This indicates bookkeeping bug.\n")
            raise HalotoolsError(msg)

        try:
            host_halo_bin_numbers = host_halo_table[self.host_halo_binning_key].data
        except KeyError:
            msg = ("The input ``halo_table`` is missing the ``{0}`` key.\n"
                "This indicates that the ``preprocess_subhalo_table`` method \n"
                "was not called prior to calling the ``inherit_subhalo_properties`` method.\n")
            raise HalotoolsError(msg.format(self.host_halo_binning_key))

        satellite_selection_idx, missing_subhalo_mask = calculate_satellite_selection_mask(
            subhalo_table['_subhalo_inheritance_id'].data, occupations,
            host_halo_table['_subhalo_inheritance_id'].data, host_halo_bin_numbers,
            fill_remaining_satellites=True, seed=seed)

        return satellite_selection_idx, missing_subhalo_mask

    def inherit_subhalo_properties(self, seed=None, **kwargs):
        """
        """
        subhalo_table = kwargs['subhalo_table']
        host_halo_table = kwargs['halo_table']
        occupations = kwargs['_occupation'][self.gal_type]
        Lbox = kwargs['Lbox']

        satellite_selection_idx, missing_subhalo_mask = self._retrieve_satellite_selection_idx(
            host_halo_table, subhalo_table, occupations, seed=seed)

        galaxy_table = kwargs['table']
        try:
            assert len(galaxy_table) == occupations.sum()
        except AssertionError:
            msg = ("The input `occupations`` array has an inconsistent total sum\n"
                "with the length of the input ``galaxy_table`` slice.\n"
                "This indicates a bookkeeping bug.\n")
            raise HalotoolsError(msg)

        galaxy_table['real_subhalo'][~missing_subhalo_mask] = True

        self._inherit_props_from_true_subhalos(galaxy_table, subhalo_table,
            satellite_selection_idx, missing_subhalo_mask)

        self._inherit_props_for_remaining_satellites(galaxy_table, subhalo_table,
            satellite_selection_idx, missing_subhalo_mask, Lbox)

    def _inherit_props_from_true_subhalos(self, galaxy_table, subhalo_table,
            satellite_selection_idx, missing_subhalo_mask):
        """
        """
        for subhalo_table_key, value in self.inherited_subhalo_props_dict.items():
            galaxy_table_key = value[0]
            galaxy_table[galaxy_table_key][~missing_subhalo_mask] = (
                subhalo_table[subhalo_table_key][satellite_selection_idx][~missing_subhalo_mask])

    def _inherit_props_for_remaining_satellites(self, galaxy_table, subhalo_table,
            satellite_selection_idx, missing_subhalo_mask, Lbox):
        """
        """
        poskeys = ('x', 'y', 'z')
        for axis,poskey in enumerate(poskeys):
            subhalo_hostpos_key = 'halo_' + poskey + '_host_halo'
            subhalo_poskey = 'halo_' + poskey
            subhalo_hostpos = subhalo_table[subhalo_hostpos_key][satellite_selection_idx][missing_subhalo_mask]
            subhalo_pos = subhalo_table[subhalo_poskey][satellite_selection_idx][missing_subhalo_mask]

            subhalo_hostvel_key = 'halo_v' + poskey + '_host_halo'
            subhalo_velkey = 'halo_v' + poskey
            subhalo_hostvel = subhalo_table[subhalo_hostvel_key][satellite_selection_idx][missing_subhalo_mask]
            subhalo_vel = subhalo_table[subhalo_velkey][satellite_selection_idx][missing_subhalo_mask]

            s, pbc_correction = _sign_pbc(subhalo_pos, subhalo_hostpos,
                period=Lbox[axis], return_pbc_correction=True)
            absd = np.abs(subhalo_pos - subhalo_hostpos)
            relpos = s*np.where(absd > Lbox[axis]/2., Lbox[axis] - absd, absd)
            relvel = pbc_correction*(subhalo_vel-subhalo_hostvel)
            galaxy_table[poskey][missing_subhalo_mask] += relpos
            galaxy_table['v'+poskey][missing_subhalo_mask] += relvel

        for subhalo_table_key, value in self.inherited_subhalo_props_dict.items():
            galaxy_table_key = value[0]
            if galaxy_table_key not in ('x', 'y', 'z', 'vx', 'vy', 'vz'):
                galaxy_table[galaxy_table_key][missing_subhalo_mask] = (
                    subhalo_table[subhalo_table_key][satellite_selection_idx][missing_subhalo_mask])

    def preprocess_subhalo_table(self, host_halo_table, subhalo_table):
        """ Method makes cuts and organizes the memory layout of the input
        ``subhalo_table`` and ``host_halo_table``.

        The only subhalos that will be kept are those with a value
        in their ``halo_hostid`` column that matches an
        entry of the ``halo_id`` column of the input ``host_halo_table``.
        The returned subhalo table will be sorted according to
        ``self.subhalo_table_sorting_keys``.
        The returned ``host_halo_table`` will be sorted by the first two
        entries of ``subhalo_table_sorting_keys``.

        Parameters
        ----------
        host_halo_table = table

        subhalo_table = table

        Returns
        --------
        processed_subhalo_table : table
            Astropy Table of subhalos with a matching host_halo,
            sorted according to ``self.subhalo_table_sorting_keys``.


        Notes
        -----
        The requirement of having a matching host halo is the basis of the algorithm,
        and this is implemented by requiring that each subhalo's ``halo_hostid``
        column has a matching value in the ``halo_id`` column of the input.
        The reason that such subhalos could exist at all is if they are part of a
        larger system that is just merging into a still larger host halo.
        In such a case, the ``halo_upid`` of a sub-subhalo may point to
        the ``halo_id`` of a subhalo that has *just* fallen into a larger host,
        so that the center of the sub-subhalo has not quite passed within the
        virial radius of its ultimate host. These subhalos are very rare,
        and are thrown out to make bookkeeping simpler.
        """
        host_halo_table.sort(self.binning_key)
        host_halo_table['_subhalo_inheritance_id'] = np.arange(len(host_halo_table))

        idxA, idxB = crossmatch(subhalo_table['halo_hostid'].data,
            host_halo_table['halo_id'].data)
        subs_with_matching_hosts = subhalo_table[idxA]

        subs_with_matching_hosts['_subhalo_inheritance_id'] = (
            host_halo_table['_subhalo_inheritance_id'][idxB])

        if self.reverse_intra_halo_order:
            subs_with_matching_hosts[self.subhalo_table_sorting_keys[-1]][:] *= -1
            subs_with_matching_hosts.sort(self.subhalo_table_sorting_keys)
            subs_with_matching_hosts[self.subhalo_table_sorting_keys[-1]][:] *= -1
        else:
            subs_with_matching_hosts.sort(self.subhalo_table_sorting_keys)

        if not array_is_monotonic(subs_with_matching_hosts['_subhalo_inheritance_id'].data,
                strict=False):
            raise HalotoolsError(sorting_error_msg)

        host_halo_prop = host_halo_table[self.binning_key].data
        bins = self.host_haloprop_bins
        host_halo_bin_numbers = np.digitize(host_halo_prop, bins).astype(int)
        self.host_halo_binning_key = self.binning_key + '_bin_number'
        host_halo_table[self.host_halo_binning_key] = host_halo_bin_numbers

        phase_space_keys = ('halo_x', 'halo_y', 'halo_z',
            'halo_vx', 'halo_vy', 'halo_vz')
        subs_with_matching_hosts[self.host_halo_binning_key] = 0
        for key in phase_space_keys:
            subs_with_matching_hosts[key+'_host_halo'] = 0.
        idxA, idxB = crossmatch(subs_with_matching_hosts['halo_hostid'].data,
            host_halo_table['halo_id'].data)
        subs_with_matching_hosts[self.host_halo_binning_key][idxA] = (
            host_halo_table[self.host_halo_binning_key][idxB])
        for key in phase_space_keys:
            subs_with_matching_hosts[key+'_host_halo'][idxA] = host_halo_table[key][idxB]

        self._check_bins_satisfy_requirements(
            host_halo_table[self.binning_key].data,
            subs_with_matching_hosts[self.binning_key].data,
            subs_with_matching_hosts[self.host_halo_binning_key].data,
            self.host_haloprop_bins)

        return host_halo_table, subs_with_matching_hosts

    def _check_bins_satisfy_requirements(self, host_halo_prop, subhalo_prop,
            subhalo_bin_numbers, bins):
        """
        """
        binning_key = self.subhalo_table_sorting_keys[0]
        try:
            assert host_halo_prop[0] > bins[0]
            assert host_halo_prop[-1] < bins[-1]
        except AssertionError:
            msg = ("The model ``host_haloprop_bins`` spans the range "
                "({0}, {1}).\n But the range required by "
                "the halo_table is ({2}, {3}).\n")
            raise ValueError(msg.format(bins[0], bins[1], host_halo_prop[0], host_halo_prop[-1]))

        try:
            assert host_halo_prop[0] <= subhalo_prop[0]
            assert host_halo_prop[-1] >= subhalo_prop[-1]
        except AssertionError:
            msg = ("The ``{0}`` column of the input ``subhalo_table`` "
                "has not been calculated properly.\n"
                "For the ``host_halo_table``, this property spans the range ({1}, {2}).\n"
                "For the ``subhalo_table``, this property spans the range ({3}, {4}).\n"
                "Since the ``subhalo_table`` has already been culled of subs without a matching host\n"
                "the fact that the ``subhalo_table`` spans a broader range can only mean\n"
                "that one or more values of the {5} key in the ``subhalo_table``\n"
                "has not been calculated correctly.\n".format(
                    binning_key, host_halo_prop[0], host_halo_prop[-1],
                    subhalo_prop[0], subhalo_prop[-1], binning_key))
            raise ValueError(msg)

        subhalo_counts = np.histogram(subhalo_bin_numbers, np.arange(1, len(bins)))[0]
        try:
            assert np.all(subhalo_counts > 1)
        except AssertionError:
            msg = "There must be at least 1 subhalo in each bin of {0}".format(binning_key)
            raise ValueError(msg)


def _sign_pbc(x1, x2, period=None, equality_fill_val=0., return_pbc_correction=False):
    """ Exact copy-and-paste of the mock.observables.catalog_analysis_helpers.sign_pbc
    function, reimplemented here to circumvent a mysterious namespace collision that only
    occurs during particular modes of unit-testing. Strictly for internal use.
    """
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    result = np.sign(x1 - x2)

    if period is not None:
        try:
            assert np.all(x1 >= 0)
            assert np.all(x2 >= 0)
            assert np.all(x1 <= period)
            assert np.all(x2 <= period)
        except AssertionError:
            msg = "If period is not None, all values of x and y must be between [0, period)"
            raise ValueError(msg)

        d = np.abs(x1-x2)
        pbc_correction = np.sign(period/2. - d)
        result = pbc_correction*result

    if equality_fill_val != 0:
        result = np.where(result == 0, equality_fill_val, result)

    if return_pbc_correction:
        return result, pbc_correction
    else:
        return result
