# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
"""
spatial is a sub-package designed to calculate spatial quantities. 
"""

__all__=['geometry', 'distances', 'kdtrees']

from .geometry import *
from .distances import *