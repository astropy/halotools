""" Sub-package of `~halotools.mock_observables` providing  
calculations of pairwise velocity statistics. 
"""
from __future__ import absolute_import

__all__ = ('mean_radial_velocity_vs_r', 'radial_pvd_vs_r',
    'mean_los_velocity_vs_rp', 'los_pvd_vs_rp')

from .pairwise_velocity_stats import mean_radial_velocity_vs_r
from .pairwise_velocity_stats import radial_pvd_vs_r
from .pairwise_velocity_stats import mean_los_velocity_vs_rp
from .pairwise_velocity_stats import los_pvd_vs_rp
