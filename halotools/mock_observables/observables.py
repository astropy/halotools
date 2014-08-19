""" 
Functions that compute statistics of a mock galaxy catalog in a periodic box. 
Still largely unused in its present form, and needs to be integrated with 
the pair counter and subvolume membership methods.
"""

#from __future__ import (absolute_import, division, print_function,
#                        unicode_literals)

__all__=['two_point_correlation_function']

import numpy as np
from math import pi


def two_point_correlation_function(sample1, rbins, sample2 = None, Lbox=None, max_sample_size=int(1e4)):
    """ Place-holder function for the two-point function. 

    Parameters 
    ----------
    sample1 : array_like
        Npts x 3 numpy array containing 3d positions of Npts. 

    rbins : array_like
        numpy array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.

    sample2 : array_like, optional
        Npts x 3 numpy array containing 3d positions of Npts.

    Lbox : float
        Size of the simulation. Defines periodic boundary conditioning.

    max_sample_size : int, optional
        Defines maximum size of the sample that will be passed to the KDtree pair counter. 

        If sample size exeeds max_sample_size, the sample will be randomly down-sampled 
        such that the subsamples are (roughly) equal to max_sample_size. 
        Subsamples will be passed to the pair counter in a simple loop, 
        and the correlation function will be estimated from the median pair counts in each bin. 

    Returns 
    -------
    correlation_function : array_like
        array containing correlation function :math:`\\xi` computed in each of the Nrbins 
        defined by input `rbins`.

        :math:`1 + \\xi(r) \equiv DD / RR`, 
        where `DD` is calculated by the pair counter, 
        and RR is counted by the internally defined `randoms` function.

        If sample2 is passed as input, three arrays of length Nrbins are returned: two for each of the 
        auto-correlation functions, and one for the cross-correlation function. 

    """
    sample1 = np.array(sample1)
    rbins = np.array(rbins)

    def random_pair_counts(Npts1, lower_bin_boundary, upper_bin_boundary, Lbox, Npts2 = None):
        global_volume = Lbox**3
        spherical_annulus_volume = (4.*pi/3.)*(
            upper_bin_boundary**3 - lower_bin_boundary**3)
        mean_Npts1_in_spherical_annulus = Npts1*spherical_annulus_volume/global_volume

        if Npts2 is None:
            random_pair_counts = 0.5*mean_Npts1_in_spherical_annulus*(
                mean_Npts1_in_spherical_annulus-1)
        else:
            mean_Npts2_in_spherical_annulus = Npts2*spherical_annulus_volume/global_volume
            random_pair_counts = (
                mean_Npts1_in_spherical_annulus*mean_Npts2_in_spherical_annulus)

        return random_pair_counts


    pass







