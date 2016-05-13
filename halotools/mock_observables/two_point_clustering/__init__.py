""" Subpackage containing modules of functions that calculate many variations 
on galaxy/halo clustering, e.g., three-dimensional clustering 
`~halotools.mock_observables.tpcf`, projected clustering `~halotools.mock_observables.wp`, 
RSD multipoles `~halotools.mock_observables.tpcf_multipole`, 
galaxy-galaxy lensing `~halotools.mock_observables.delta_sigma`, and more. 
"""
from __future__ import absolute_import 

__all__ = ('angular_tpcf', )

from .angular_tpcf import angular_tpcf 
