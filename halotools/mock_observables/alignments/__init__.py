"""
Subpackage containing modules of functions that calculate many variations on galaxy/halo alignments.
"""
from __future__ import absolute_import

# ellipticity-ellipticity tpcf
from .ee_3d import ee_3d
from .ee_projected import ee_projected

# ellipticity-direction tpcf
from .ed_3d import ed_3d
from .ed_projected import ed_projected

# gravitational shear-intrinsic ellipticity tpcf
from .gi_plus_3d import gi_plus_3d
from .gi_plus_projected import gi_plus_projected
from .gi_minus_3d import gi_minus_3d
from .gi_minus_projected import gi_minus_projected

# intrinsic ellipticity-intrinsic ellipticity tpcf
from .ii_plus_3d import ii_plus_3d
from .ii_plus_projected import ii_plus_projected
from .ii_minus_3d import ii_minus_3d
from .ii_minus_projected import ii_minus_projected

__all__ = ('ee_3d', 'ee_projected',
           'ed_3d', 'ed_projected',
           'gi_plus_3d', 'gi_plus_projected',
           'gi_minus_3d', 'gi_minus_projected',
           'ii_plus_3d', 'ii_plus_projected',
           'ii_minus_3d', 'ii_minus_projected')
