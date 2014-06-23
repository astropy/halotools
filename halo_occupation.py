# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:52:05 2014

@author: aphearin
"""

import numpy as np
from scipy.special import erfc

def N_cen(M,Mcut=0,sigma=0):
    """Expected number of central galaxies in a halo of mass 10**M."""
    return .5*erfc(ln10*(Mcut-M)/(sqrt2*sigma))
