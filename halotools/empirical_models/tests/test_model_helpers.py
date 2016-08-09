"""
"""

import numpy as np
from unittest import TestCase
from astropy.tests.helper import pytest
from astropy.utils.misc import NumpyRNGContext

from .. import model_helpers as occuhelp
from ...custom_exceptions import HalotoolsError

__all__ = ('TestModelHelpers', 'test_custom_spline1')

fixed_seed = 43


def test_custom_spline1():
    table_abscissa = (0, 1)
    table_ordinates = (0, 1, 2)
    with pytest.raises(HalotoolsError) as err:
        __ = occuhelp.custom_spline(table_abscissa, table_ordinates)
    substr = "table_abscissa and table_ordinates must have the same length"
    assert substr in err.value.args[0]


def test_custom_spline2():
    table_abscissa = (0, 1, 2)
    table_ordinates = (0, 1, 2)
    with pytest.raises(HalotoolsError) as err:
        __ = occuhelp.custom_spline(table_abscissa, table_ordinates, k=-2)
    substr = "Spline degree must be non-negative"
    assert substr in err.value.args[0]


def test_custom_spline3():
    table_abscissa = (0, 1, 2)
    table_ordinates = (0, 1, 2)
    with pytest.raises(HalotoolsError) as err:
        __ = occuhelp.custom_spline(table_abscissa, table_ordinates, k=0)
    substr = "In spline_degree=0 edge case,"
    assert substr in err.value.args[0]


def test_create_composite_dtype():
    dt1 = np.dtype([('x', 'f4')])
    dt2 = np.dtype([('x', 'i4')])
    with pytest.raises(HalotoolsError) as err:
        result = occuhelp.create_composite_dtype([dt1, dt2])
    substr = "Inconsistent dtypes for name"
    assert substr in err.value.args[0]


def test_bind_default_kwarg_mixin_safe():

    class DummyClass(object):

        def __init__(self, d):
            self.abc = 4

    constructor_kwargs = {'abc': 10}
    obj = DummyClass(constructor_kwargs)
    keyword_argument = 'abc'
    default_value = 0

    with pytest.raises(HalotoolsError) as err:
        __ = occuhelp.bind_default_kwarg_mixin_safe(
            obj, keyword_argument, constructor_kwargs, default_value)
    substr = "Do not pass the  ``abc`` keyword argument "
    assert substr in err.value.args[0]


def test_bounds_enforcing_decorator_factory():
    """
    """
    def f(x):
        return x
    decorator = occuhelp.bounds_enforcing_decorator_factory(0, 1, warning=True)
    decorated_f = decorator(f)
    result = decorated_f(-1)
    assert result == 0


class TestModelHelpers(TestCase):
    """ Class providing tests of the `~halotools.empirical_models.model_helpers`.
    """

    def test_enforce_periodicity_of_box(self):

        box_length = 250
        Npts = int(1e5)
        with NumpyRNGContext(fixed_seed):
            coords = np.random.uniform(0, box_length, Npts*3).reshape(Npts, 3)

        perturbation_size = box_length/10.
        with NumpyRNGContext(fixed_seed):
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
