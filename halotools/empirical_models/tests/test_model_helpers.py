"""
"""
import numpy as np
import pytest
from astropy.utils.misc import NumpyRNGContext

from ..model_helpers import custom_spline, create_composite_dtype
from ..model_helpers import enforce_periodicity_of_box
from ..model_helpers import call_func_table, bind_default_kwarg_mixin_safe

from ...custom_exceptions import HalotoolsError

__all__ = ('test_enforce_periodicity_of_box', 'test_custom_spline1',
    'test_check_multiple_box_lengths', 'test_velocity_flip')

fixed_seed = 43


def _func_maker(i):
    def f(x):
        return x + i
    return f


def test_custom_spline1():
    table_abscissa = (0, 1)
    table_ordinates = (0, 1, 2)
    with pytest.raises(HalotoolsError) as err:
        __ = custom_spline(table_abscissa, table_ordinates)
    substr = "table_abscissa and table_ordinates must have the same length"
    assert substr in err.value.args[0]


def test_custom_spline2():
    table_abscissa = (0, 1, 2)
    table_ordinates = (0, 1, 2)
    with pytest.raises(HalotoolsError) as err:
        __ = custom_spline(table_abscissa, table_ordinates, k=-2)
    substr = "Spline degree must be non-negative"
    assert substr in err.value.args[0]


def test_custom_spline3():
    table_abscissa = (0, 1, 2)
    table_ordinates = (0, 1, 2)
    with pytest.raises(HalotoolsError) as err:
        __ = custom_spline(table_abscissa, table_ordinates, k=0)
    substr = "In spline_degree=0 edge case,"
    assert substr in err.value.args[0]


def test_create_composite_dtype():
    dt1 = np.dtype([('x', 'f4')])
    dt2 = np.dtype([('x', 'i4')])
    with pytest.raises(HalotoolsError) as err:
        result = create_composite_dtype([dt1, dt2])
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
        __ = bind_default_kwarg_mixin_safe(
            obj, keyword_argument, constructor_kwargs, default_value)
    substr = "Do not pass the  ``abc`` keyword argument "
    assert substr in err.value.args[0]


def test_enforce_periodicity_of_box():
    """ Verify that enforce_periodicity_of_box results in all points located
    inside [0, Lbox]
    """

    box_length = 250
    Npts = int(1e5)
    with NumpyRNGContext(fixed_seed):
        coords = np.random.uniform(0, box_length, Npts*3).reshape(Npts, 3)

    perturbation_size = box_length/10.
    with NumpyRNGContext(fixed_seed):
        coord_perturbations = np.random.uniform(
            -perturbation_size, perturbation_size, Npts*3).reshape(Npts, 3)

    coords += coord_perturbations

    newcoords = enforce_periodicity_of_box(coords, box_length)
    assert np.all(newcoords >= 0)
    assert np.all(newcoords <= box_length)


def test_check_multiple_box_lengths():
    """ Verify that enforce_periodicity_of_box function notices when the
    some points lie many box lengths beyond +/- Lbox
    """
    box_length = 250
    Npts = int(1e4)

    x = np.linspace(-2*box_length, box_length, Npts)
    with pytest.raises(HalotoolsError) as err:
        newcoords = enforce_periodicity_of_box(x, box_length,
            check_multiple_box_lengths=True)
    substr = "There is at least one input point with a coordinate less than -Lbox"
    assert substr in err.value.args[0]

    x = np.linspace(-box_length, 2.1*box_length, Npts)
    with pytest.raises(HalotoolsError) as err:
        newcoords = enforce_periodicity_of_box(x, box_length,
            check_multiple_box_lengths=True)
    substr = "There is at least one input point with a coordinate greater than 2*Lbox"
    assert substr in err.value.args[0]

    x = np.linspace(-box_length, 2*box_length, Npts)
    newcoords = enforce_periodicity_of_box(x, box_length,
        check_multiple_box_lengths=True)


def test_velocity_flip():
    """ Verify that enforce_periodicity_of_box function preserves the sign of the
    velocity for points where PBCs needed to be enforced
    """
    box_length = 250
    Npts = int(1e4)

    x = np.linspace(-0.5*box_length, 1.5*box_length, Npts)
    vx = np.ones(Npts)

    newcoords, newvel = enforce_periodicity_of_box(
        x, box_length, velocity=vx)

    assert np.all(newvel == vx)


def test_call_func_table1():

    num_conc_bins = 5
    f_table = list(_func_maker(i) for i in range(num_conc_bins))

    num_abscissa = 7
    cum_prob = np.array(list(0.1*i for i in range(num_abscissa)))

    func_idx = np.zeros(num_abscissa)
    correct_result = cum_prob
    result = call_func_table(f_table, cum_prob, func_idx)
    assert np.all(result == correct_result)


def test_call_func_table2():

    num_conc_bins = 5
    f_table = list(_func_maker(i) for i in range(num_conc_bins))

    num_abscissa = 7
    cum_prob = np.array(list(0.1*i for i in range(num_abscissa)))

    func_idx = np.zeros(num_abscissa) + 3
    func_idx[2:] = 0
    correct_result = np.zeros(num_abscissa)
    correct_result[:2] = cum_prob[:2] + 3
    correct_result[2:] = cum_prob[2:]

    result = call_func_table(f_table, cum_prob, func_idx)

    assert np.all(result == correct_result)


def test_call_func_table3():
    pass
