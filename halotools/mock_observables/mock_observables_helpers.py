""" Module containing various helper functions used to process the 
arguments of functions throughout the `~halotools.mock_observables` package. 
""" 
from warnings import warn 
import numpy as np
import multiprocessing
num_available_cores = multiprocessing.cpu_count()

__all__ = ('enforce_pbcs', 'get_num_threads', 'get_period')

def enforce_pbcs(x, y, z, period):
    """ Verify that the input sample is properly bounded in all dimensions by the input period.

    Parameters 
    -----------
    x, y, z : arrays 

    period : 3-element sequence 
    """
    try:
        assert np.all(x >= 0)
        assert np.all(y >= 0)
        assert np.all(z >= 0)
    except:
        msg = ("You set periodic boundary conditions to be True by passing in \n"
            "period = (%.2f, %.2f, %.2f), but your input data has negative values,\n"
            "indicating that you forgot to apply periodic boundary conditions.\n")
        raise ValueError(msg % (period[0], period[1], period[2]))

    try:
        assert np.all(x <= period[0])
    except:
        msg = ("You set xperiod = %.2f but there are values in the x-dimension \n"
            "of the input data that exceed this value")
        raise ValueError(msg % period[0])

    try:
        assert np.all(y <= period[1])
    except:
        msg = ("You set yperiod = %.2f but there are values in the y-dimension \n"
            "of the input data that exceed this value")
        raise ValueError(msg % period[1])

    try:
        assert np.all(z <= period[2])
    except:
        msg = ("You set zperiod = %.2f but there are values in the z-dimension \n"
            "of the input data that exceed this value")
        raise ValueError(msg % period[2])

def get_num_threads(input_num_threads, enforce_max_cores = False):
    """ Helper function requires that ``input_num_threads`` either be an 
    integer or the string ``max``. If ``input_num_threads`` exceeds the 
    number of available cores, a warning will be issued. 
    In this event,  ``enforce_max_cores`` is set to True, 
    then ``num_threads`` is automatically set to num_cores. 
    """
    if input_num_threads=='max':
        num_threads = num_available_cores
    else:
        try:
            num_threads = int(input_num_threads)
            assert num_threads == input_num_threads
        except:
            msg = ("Input ``num_threads`` must be an integer")
            raise ValueError(msg)

    if num_threads > num_available_cores:
        msg = ("Input ``num_threads`` = %i exceeds the ``num_available_cores`` = %i.\n")

        if enforce_max_cores is True:
            msg += ("Since ``enforce_max_cores`` is True, "
                "setting ``num_threads`` to ``num_available_cores``.\n")
            num_threads = num_available_cores

        warn(msg % (num_threads, num_available_cores))

    return num_threads

def get_period(period):
    """ Helper function used to process the input ``period`` argument. 
    If ``period`` is set to None, function returns period, PBCs = (None, False). 
    Otherwise, function returns ([period, period, period], True).
    """

    if period is None:
        PBCs = False
    else:
        PBCs = True
        period = np.atleast_1d(period).astype(float)

        if len(period) == 1:
            period = np.array([period[0]]*3).astype(float)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
            assert len(period) == 3
        except AssertionError:
            msg = ("Input ``period`` must be either a scalar or a 3-element sequence.\n"
                "All values must bounded positive numbers.\n")
            raise ValueError(msg)

    return period, PBCs


