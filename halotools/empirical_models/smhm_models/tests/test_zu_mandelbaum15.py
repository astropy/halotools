"""
"""
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np

from ...smhm_models import ZuMandelbaum15SmHm


__all__ = ('test_mc_scatter1', )


def test_mc_scatter1():
    model = ZuMandelbaum15SmHm(redshift=0)
    sm = model.mc_stellar_mass(prim_haloprop=1e12)
