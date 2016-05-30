#!/usr/bin/env python

import numpy as np
from unittest import TestCase
from astropy.tests.helper import pytest

from .. import model_helpers as occuhelp
from ...custom_exceptions import HalotoolsError

__all__ = ['TestModelHelpers']


class TestModelHelpers(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.model_helpers`.
    """

    def test_enforce_periodicity_of_box(self):

        box_length = 250
        Npts = int(1e5)
        coords = np.random.uniform(0, box_length, Npts*3).reshape(Npts, 3)

        perturbation_size = box_length/10.
        coord_perturbations = np.random.uniform(
            -perturbation_size, perturbation_size, Npts*3).reshape(Npts, 3)

        coords += coord_perturbations

        newcoords = occuhelp.enforce_periodicity_of_box(coords, box_length)
        assert np.all(newcoords >= 0)
        assert np.all(newcoords <= box_length)

    def test_check_multiple_box_lengths(self):
        box_length = 250
        Npts = int(1e4)

        x = np.linspace(-2*box_length, box_length, Npts)
        with pytest.raises(HalotoolsError) as err:
            newcoords = occuhelp.enforce_periodicity_of_box(x, box_length,
                check_multiple_box_lengths=True)
        substr = "There is at least one input point with a coordinate less than -Lbox"
        assert substr in err.value.args[0]

        x = np.linspace(-box_length, 2.1*box_length, Npts)
        with pytest.raises(HalotoolsError) as err:
            newcoords = occuhelp.enforce_periodicity_of_box(x, box_length,
                check_multiple_box_lengths=True)
        substr = "There is at least one input point with a coordinate greater than 2*Lbox"
        assert substr in err.value.args[0]

        x = np.linspace(-box_length, 2*box_length, Npts)
        newcoords = occuhelp.enforce_periodicity_of_box(x, box_length,
            check_multiple_box_lengths=True)

    def test_velocity_flip(self):
        box_length = 250
        Npts = int(1e4)

        x = np.linspace(-0.5*box_length, 1.5*box_length, Npts)
        vx = np.ones(Npts)

        newcoords, newvel = occuhelp.enforce_periodicity_of_box(
            x, box_length, velocity=vx)

        inbox = ((x >= 0) & (x <= box_length))
        assert np.all(newvel[inbox] == 1.0)
        assert np.all(newvel[~inbox] == -1.0)
