#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

from unittest import TestCase
import pytest 

import numpy as np 
from copy import copy 

from .. import FlatRectanguloidTree

__all__ = ['TestFlatRectanguloidTree']

class TestFlatRectanguloidTree(TestCase):
    """ Class providing tests of the `~halotools.mock_observables.pair_counters.FlatRectanguloidDoubleTree`. 
    """

    def setup_class(self):
        """ Pre-load various arrays into memory for use by all tests. 
        """
        Npts, Lbox = 1e4, 1000
        xperiod, yperiod, zperiod = Lbox, Lbox, Lbox

        approx_xcell_size1 = Lbox/10.
        approx_ycell_size1 = Lbox/10.
        approx_zcell_size1 = Lbox/10.

        approx_xcell_size2 = Lbox/5.
        approx_ycell_size2 = Lbox/10.
        approx_zcell_size2 = Lbox/20.

        np.random.seed(43)
        x = np.random.uniform(0, Lbox, Npts)
        y = np.random.uniform(0, Lbox, Npts) 
        z = np.random.uniform(0, Lbox, Npts) 

        self.tree1 = FlatRectanguloidTree(x, y, z, 
            approx_xcell_size1, approx_ycell_size1, approx_zcell_size1, 
            xperiod, yperiod, zperiod)

        self.tree2 = FlatRectanguloidTree(x, y, z, 
            approx_xcell_size2, approx_ycell_size2, approx_zcell_size2, 
            xperiod, yperiod, zperiod)

    def test_slices(self):

        i = 13
        ith_subvol_slice = self.tree1.slice_array[i]
        xcoords_ith_subvol = self.tree1.x[ith_subvol_slice]
        ycoords_ith_subvol = self.tree1.y[ith_subvol_slice]
        zcoords_ith_subvol = self.tree1.z[ith_subvol_slice]

        assert np.all(xcoords_ith_subvol <= 100.)
        assert np.all(0 <= xcoords_ith_subvol)
        assert np.all(ycoords_ith_subvol <= 200.)
        assert np.all(100. <= ycoords_ith_subvol)
        assert np.all(zcoords_ith_subvol <= 400.)
        assert np.all(300. <= zcoords_ith_subvol)

        i = 1
        ith_subvol_slice = self.tree2.slice_array[i]
        xcoords_ith_subvol = self.tree2.x[ith_subvol_slice]
        ycoords_ith_subvol = self.tree2.y[ith_subvol_slice]
        zcoords_ith_subvol = self.tree2.z[ith_subvol_slice]

        assert np.all(xcoords_ith_subvol <= 200.)
        assert np.all(0 <= xcoords_ith_subvol)


        assert np.all(ycoords_ith_subvol <= 100.)
        assert np.all(0 <= ycoords_ith_subvol)

        assert np.all(zcoords_ith_subvol <= 100.)
        assert np.all(50. <= zcoords_ith_subvol)









