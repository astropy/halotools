""" Module containing helper functions used to process the arguments 
passed to functions in the `~halotools.mock_observables.radial_profiles` sub-package.
""" 
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

__all__ = ('bounds_check_sample2_quantity', )

def bounds_check_sample2_quantity(sample2, sample2_quantity):
    """ Function enforces that input ``sample2_quantity`` has the appropriate shape.
    """
    npts2 = sample2.shape[0]
    sample2_quantity = np.atleast_1d(sample2_quantity)
    npts_quantity2 = len(sample2_quantity)
    try:
        assert npts_quantity2 == npts2 
    except AssertionError:
        msg = ("Input ``sample2_quantity`` has %i elements, "
            "but input ``sample2`` has %i elements.\n" % (npts_quantity2, npts2))
        raise ValueError(msg)
    return sample2_quantity

def get_distance_normalization(sample1, normalize_rbins_by):
    """ Function enforces that input ``normalize_rbins_by`` has the appropriate shape 
    and returns the appropriate ``distance_normalization`` array, 
    including proper handling of the default None argument. 
    """
    npts1 = sample1.shape[0]
    if normalize_rbins_by is None:
        distance_normalization = np.ones(npts1)
    else:
        distance_normalization = np.atleast_1d(normalize_rbins_by)
        npts_normalization = len(distance_normalization)
        try:
            assert npts_normalization == npts1
        except AssertionError:
            msg = ("Input ``normalize_rbins_by`` has %i elements, "
                "but input ``sample1`` has %i elements.\n" % (npts_normalization, npts1))
            raise ValueError(msg)




