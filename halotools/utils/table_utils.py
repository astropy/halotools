r"""
Modules performing small, commonly used tasks throughout the package.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from math import ceil
import numpy as np
from warnings import warn
from astropy.table import Table

from ..custom_exceptions import HalotoolsError

__all__ = ['SampleSelector']


def compute_conditional_percentiles(**kwargs):
    r"""
    In bins of the ``prim_haloprop``, compute the rank-order percentile
    of the input ``table`` based on the value of ``sec_haloprop``.

    Parameters
    ----------
    table : astropy table, optional
        a keyword argument that stores halo catalog being used to make mock galaxy population
        If a `table` is passed, the `prim_haloprop_key` and `sec_haloprop_key` keys
        must also be passed. If not passing a `table`, you must directly pass the
        `prim_haloprop` and `sec_haloprop` keyword arguments.

    prim_haloprop_key : string, optional
        Name of the column of the input ``table`` that will be used to access the
        primary halo property. `compute_conditional_percentiles` bins the ``table`` by
        ``prim_haloprop_key`` when computing the result.

    sec_haloprop_key : string, optional
        Name of the column of the input ``table`` that will be used to access the
        secondary halo property. `compute_conditional_percentiles` bins the ``table`` by
        ``prim_haloprop_key``, and in each bin uses the value stored in ``sec_haloprop_key``
        to compute the ``prim_haloprop``-conditioned rank-order percentile.

    prim_haloprop : array_like, optional
        Array storing the primary halo property used to bin the input points.
        If a `prim_haloprop` is passed, you must also pass a `sec_haloprop`.

    sec_haloprop : array_like, optional
        Array storing the secondary halo property used to define the conditional percentiles
        in each bin of `prim_haloprop`.

    prim_haloprop_bin_boundaries : array, optional
        Array defining the boundaries by which we will bin the input ``table``.
        Default is None, in which case the binning will be automatically determined using
        the ``dlog10_prim_haloprop`` keyword.

    dlog10_prim_haloprop : float, optional
        Logarithmic spacing of bins of the mass-like variable within which
        we will assign secondary property percentiles. Default is 0.2.

    Examples
    --------
    >>> from halotools.sim_manager import FakeSim
    >>> fakesim = FakeSim()
    >>> result = compute_conditional_percentiles(table = fakesim.halo_table, prim_haloprop_key = 'halo_mvir', sec_haloprop_key = 'halo_vmax')


    Notes
    -----
    The sign of the result is such that in bins of the primary property,
    *smaller* values of the secondary property
    receive *smaller* values of the returned percentile.

    """

    if 'table' in kwargs:
        table = kwargs['table']
        try:
            prim_haloprop_key = kwargs['prim_haloprop_key']
            prim_haloprop = table[prim_haloprop_key]
            sec_haloprop_key = kwargs['sec_haloprop_key']
            sec_haloprop = table[sec_haloprop_key]
        except KeyError:
            msg = ("\nWhen passing an input ``table`` to the ``compute_conditional_percentiles`` method,\n"
                "you must also pass ``prim_haloprop_key`` and ``sec_haloprop_key`` keyword arguments\n"
                "whose values are column keys of the input ``table``\n")
            raise HalotoolsError(msg)
    else:
        try:
            prim_haloprop = kwargs['prim_haloprop']
            sec_haloprop = kwargs['sec_haloprop']
        except KeyError:
            msg = ("\nIf not passing an input ``table`` to the ``compute_conditional_percentiles`` method,\n"
                "you must pass a ``prim_haloprop`` and ``sec_haloprop`` arguments\n")
            raise HalotoolsError(msg)

    def compute_prim_haloprop_bins(dlog10_prim_haloprop=0.05, **kwargs):
        r"""
        Parameters
        ----------
        prim_haloprop : array
            Array storing the value of the primary halo property column of the ``table``
            passed to ``compute_conditional_percentiles``.

        prim_haloprop_bin_boundaries : array, optional
            Array defining the boundaries by which we will bin the input ``table``.
            Default is None, in which case the binning will be automatically determined using
            the ``dlog10_prim_haloprop`` keyword.

        dlog10_prim_haloprop : float, optional
            Logarithmic spacing of bins of the mass-like variable within which
            we will assign secondary property percentiles. Default is 0.2.

        Returns
        --------
        output : array
            Numpy array of integers storing the bin index of the prim_haloprop bin
            to which each halo in the input table was assigned.

        """
        try:
            prim_haloprop = kwargs['prim_haloprop']
        except KeyError:
            msg = ("The ``compute_prim_haloprop_bins`` method "
                "requires the ``prim_haloprop`` keyword argument")
            raise HalotoolsError(msg)

        try:
            prim_haloprop_bin_boundaries = kwargs['prim_haloprop_bin_boundaries']
        except KeyError:
            lg10_min_prim_haloprop = np.log10(np.min(prim_haloprop))-0.001
            lg10_max_prim_haloprop = np.log10(np.max(prim_haloprop))+0.001
            num_prim_haloprop_bins = (lg10_max_prim_haloprop-lg10_min_prim_haloprop)/dlog10_prim_haloprop
            prim_haloprop_bin_boundaries = np.logspace(
                lg10_min_prim_haloprop, lg10_max_prim_haloprop,
                num=int(ceil(num_prim_haloprop_bins)))

        # digitize the masses so that we can access them bin-wise
        output = np.digitize(prim_haloprop, prim_haloprop_bin_boundaries)

        # Use the largest bin for any points larger than the largest bin boundary,
        # and raise a warning if such points are found
        Nbins = len(prim_haloprop_bin_boundaries)
        if Nbins in output:
            msg = ("\n\nThe ``compute_prim_haloprop_bins`` function detected points in the \n"
                "input array of primary halo property that were larger than the largest value\n"
                "of the input ``prim_haloprop_bin_boundaries``. All such points will be assigned\n"
                "to the largest bin.\nBe sure that this is the behavior you expect for your application.\n\n")
            warn(msg)
            output = np.where(output == Nbins, Nbins-1, output)

        return output

    compute_prim_haloprop_bins_dict = {}
    compute_prim_haloprop_bins_dict['prim_haloprop'] = prim_haloprop
    try:
        compute_prim_haloprop_bins_dict['prim_haloprop_bin_boundaries'] = (
            kwargs['prim_haloprop_bin_boundaries'])
    except KeyError:
        pass
    try:
        compute_prim_haloprop_bins_dict['dlog10_prim_haloprop'] = kwargs['dlog10_prim_haloprop']
    except KeyError:
        pass
    prim_haloprop_bins = compute_prim_haloprop_bins(**compute_prim_haloprop_bins_dict)

    output = np.zeros_like(prim_haloprop)

    # sort on secondary property only with each mass bin
    bins_in_halocat = set(prim_haloprop_bins)
    for ibin in bins_in_halocat:
        indices_of_prim_haloprop_bin = np.where(prim_haloprop_bins == ibin)[0]

        num_in_bin = len(sec_haloprop[indices_of_prim_haloprop_bin])

        # Find the indices that sort by the secondary property
        ind_sorted = np.argsort(sec_haloprop[indices_of_prim_haloprop_bin])

        percentiles = np.zeros(num_in_bin)
        percentiles[ind_sorted] = (np.arange(num_in_bin) + 1.0) / float(num_in_bin)

        # place the percentiles into the catalog
        output[indices_of_prim_haloprop_bin] = percentiles

    return output


class SampleSelector(object):
    r""" Container class for commonly used sample selections.
    """

    @staticmethod
    def host_halo_selection(return_subhalos=False, **kwargs):
        r""" Method divides sample in to host halos and subhalos, and returns
        either the hosts or the hosts and the subs depending
        on the value of the input ``return_subhalos``.
        """
        table = kwargs['table']
        mask = table['halo_upid'] == -1
        if return_subhalos is False:
            return table[mask]
        else:
            return table[mask], table[~mask]

    @staticmethod
    def property_range(lower_bound=-float("inf"), upper_bound=float("inf"),
            return_complement=False, host_halos_only=False, subhalos_only=False, **kwargs):
        r""" Method makes a cut on an input table column based on an input upper and lower bound, and
        returns the cut table.

        Parameters
        ----------
        table : Astropy Table object, keyword argument

        key : string, keyword argument
            Column name that will be used to apply the cut

        lower_bound : float, optional keyword argument
            Minimum value for the input column of the returned table. Default is :math:`-\infty`.

        upper_bound : float, optional keyword argument
            Maximum value for the input column of the returned table. Default is :math:`+\infty`.

        return_complement : bool, optional keyword argument
            If True, `property_range` gives the table elements that do not pass the cut
            as the second return argument. Default is False.

        host_halos_only : bool, optional keyword argument
            If true, `property_range` will use the `host_halo_selection` method to
            make an additional cut on the sample so that only host halos are returned.
            Default is False

        subhalos_only : bool, optional keyword argument
            If true, `property_range` will use the `host_halo_selection` method to
            make an additional cut on the sample so that only subhalos are returned.
            Default is False


        Returns
        -------
        cut_table : Astropy Table object

        Examples
        ---------
        To demonstrate the `property_range` method, we will start out by loading
        a table of halos into memory using the `FakeSim` class:

        >>> from halotools.sim_manager import FakeSim
        >>> halocat = FakeSim()
        >>> halos = halocat.halo_table

        To make a cut on the halo catalog to select halos in a specific mass range:

        >>> halo_sample = SampleSelector.property_range(table = halos, key = 'halo_mvir', lower_bound = 1e12, upper_bound = 1e13)

        To apply this same cut, and also only select host halos passing the cut, we use the ``host_halos_only`` keyword:

        >>> host_halo_sample = SampleSelector.property_range(table = halos, key = 'halo_mvir', lower_bound = 1e12, upper_bound = 1e13, host_halos_only=True)

        The same applies if we only want subhalos returned only now we use the ``subhalos_only`` keyword:

        >>> subhalo_sample = SampleSelector.property_range(table = halos, key = 'halo_mvir', lower_bound = 1e12, upper_bound = 1e13, subhalos_only=True)

        """
        table = kwargs['table']

        # First apply the host halo cut, if applicable
        if (host_halos_only is True) & (subhalos_only is True):
            raise KeyError("You cannot simultaneously select only host halos and only subhalos")
        elif host_halos_only is True:
            table = SampleSelector.host_halo_selection(table=table)
        elif subhalos_only is True:
            hosts, table = SampleSelector.host_halo_selection(table=table, return_subhalos=True)

        key = kwargs['key']
        mask = (table[key] >= lower_bound) & (table[key] <= upper_bound)

        if return_complement is True:
            return table[mask], table[np.invert(mask)]
        else:
            return table[mask]

    @staticmethod
    def split_sample(**kwargs):
        r""" Method divides a sample into subsamples based on the percentile ranking of a given property.

        Parameters
        ----------
        table : Astropy Table object, keyword argument

        key : string, keyword argument
            Column name that will be used to define the percentiles

        percentiles : array_like
            Sequence of percentiles used to define the returned subsamples. If ``percentiles``
            has more than one element, the elements must be monotonically increasing.
            If ``percentiles`` is length-N, there will be N+1 returned subsamples.

        Returns
        -------
        subsamples : list

        Examples
        --------
        To demonstrate the `split_sample` method, we will start out by loading
        a table of halos into memory using the `FakeSim` class:

        >>> from halotools.sim_manager import FakeSim
        >>> halocat = FakeSim()
        >>> halos = halocat.halo_table

        We can easily use `split_sample` to divide the sample into a high-Vmax and low-Vmax subsamples:

        >>> sample_below_median, sample_above_median = SampleSelector.split_sample(table = halos, key = 'halo_vmax', percentiles = 0.5)

        Likewise, we can do the same thing to divide the sample into quartiles:

        >>> lowest, lower, higher, highest = SampleSelector.split_sample(table = halos, key = 'halo_zhalf', percentiles = [0.25, 0.5, 0.75])

        The following alternative syntax is also supported:

        >>> subsample_collection = SampleSelector.split_sample(table = halos, key = 'halo_zhalf', percentiles = [0.25, 0.5, 0.75])
        >>> lowest, lower, higher, highest = subsample_collection

        """
        table = kwargs['table']
        if not isinstance(table, Table):
            raise TypeError("Input table must be an Astropy Table instance")

        key = kwargs['key']
        if key not in list(table.keys()):
            raise KeyError("The ``{0}`` key does not appear in the table you are trying \n"
                "to split into subsamples".format(key))
        table.sort(key)

        percentiles = kwargs['percentiles']
        percentiles = np.array(percentiles)
        if np.shape(percentiles) == ():
            percentiles = np.array([percentiles])
        num_total = len(table)
        if len(percentiles) >= num_total:
            raise ValueError("Input length of percentiles must be less than input table length")

        indices = percentiles*num_total
        indices = np.insert(indices, 0, 0)
        percentiles = np.insert(percentiles, 0, 0)
        indices = indices.astype(int)
        indices = np.append(indices, len(table))
        percentiles = np.append(percentiles, 1.0)

        d = np.diff(indices)
        d[-1] -= 1
        if 0 in d:
            print("Raise exception: too many percentile bins")
            idx_too_few = np.nanargmin(d)
            raise ValueError("The input percentiles spacing is too fine.\n"
                "For example, there are no table elements in the percentile range (%.2f, %.2f)" %
                  (percentiles[idx_too_few], percentiles[idx_too_few+1]))

        result = np.zeros(len(indices)-1, dtype=object)
        for i, first_idx, last_idx in zip(list(range(len(result))), indices[:-1], indices[1:]):
            result[i] = table[first_idx:last_idx]

        return result
