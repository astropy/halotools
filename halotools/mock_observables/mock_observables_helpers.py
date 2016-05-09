""" Module containing various helper functions used to process the 
arguments of functions throughout the `~halotools.mock_observables` package. 
""" 

import numpy as np

__all__ = ('enforce_pbcs', )

def enforce_pbcs(x, y, z, period):
    """ Verify that the input sample is properly bounded in all dimensions by the input period. 
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



