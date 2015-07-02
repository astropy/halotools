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
    def property_range(lower_bound = -float("inf"), upper_bound = float("inf"), 
        return_complement = False, **kwargs):
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

        Returns 
        -------
        cut_table : Astropy Table object

        Examples 
        ---------
        To demonstrate the `property_range` method, we will start out by loading 
        a table of halos into memory using the `FakeSim` class:

        >>> snapshot = FakeSim()
        >>> halos = snapshot.halos





        """
        table = kwargs['table']
        key = kwargs['key']
        mask = (table[key] > lower_bound) & (table[key] < upper_bound)

        if return_complement is True:
            return table[mask], table[np.invert(mask)]
        else:
            return table[mask]














