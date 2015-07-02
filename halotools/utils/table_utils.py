# -*- coding: utf-8 -*-
"""

Modules performing small, commonly used tasks throughout the package.

"""

__all__ = ['SampleSelector']

import numpy as np
import collections
from ..sim_manager.generate_random_sim import FakeSim

class SampleSelector(object):
    """ Container class for commonly used sample selections. 
    """

    @staticmethod
    def host_halo_selection(return_subhalos=False, **kwargs):
        """ Method divides sample in to host halos and subhalos, and returns 
        either the hosts or the hosts and the subs depending 
        on the value of the input ``return_subhalos``. 
        """
        table = kwargs['table']
        mask = table['upid'] == -1
        if return_subhalos is False:
            return table[mask]
        else:
            return table[mask], table[np.invert(mask)]

    @staticmethod
    def property_range(lower_bound = -float("inf"), upper_bound = float("inf"), 
        return_complement = False, host_halos_only=False, subhalos_only=False, **kwargs):
        """ Method makes a cut on an input table column based on an input upper and lower bound, and 
        returns the cut table. 

        Parameters 
        ----------
        table : Astropy Table object, keyword argument 

        key : string, keyword argument 
            Column name that will be used to apply the cut

        lower_bound : float, optional keyword argument 
            Minimum value for the input column of the returned table. Default is :math:`-\\infty`. 

        upper_bound : float, optional keyword argument 
            Maximum value for the input column of the returned table. Default is :math:`+\\infty`. 

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

        >>> snapshot = FakeSim()
        >>> halos = snapshot.halos

        To make a cut on the halo catalog to select halos in a specific mass range:

        >>> halo_sample = SampleSelector.property_range(table = halos, key = 'mvir', lower_bound = 1e12, upper_bound = 1e13)

        To apply this same cut, and also only select host halos passing the cut, we use the ``host_halos_only`` keyword:

        >>> host_halo_sample = SampleSelector.property_range(table = halos, key = 'mvir', lower_bound = 1e12, upper_bound = 1e13, host_halos_only=True)

        The same applies if we only want subhalos returned only now we use the ``subhalos_only`` keyword:

        >>> subhalo_sample = SampleSelector.property_range(table = halos, key = 'mvir', lower_bound = 1e12, upper_bound = 1e13, subhalos_only=True)

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

















