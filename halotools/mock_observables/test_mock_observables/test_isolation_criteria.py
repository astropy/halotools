#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
from ..isolation_criteria import isolatoion_criterion

__all__ = ['test_isolation_criterion_API']

####isolation criteria####################################################################
def test_isolation_criterion_API():
    
    pass
    """
    #define isolation function. This one works with magnitudes, to find galaxies with no 
    #neighbors brighter than host+0.5 mag
    def is_isolated(candidate_prop,other_prop):
        delta = 0.5
        return other_prop>(candidate_prop+delta)
    
    iso_crit = isolatoion_criterion(volume=geometry.sphere, test_func=is_isolated)
    
    from halotools import make_mocks
    mock = make_mocks.HOD_mock()
    mock.populate()
    
    result = iso_crit.apply_criterion(mock,[0])
    print(result)
    assert True==False
    """
##########################################################################################
    
    