"""
"""
import numpy as np
from astropy.table import Table
from astropy.utils.misc import NumpyRNGContext

from ...occupation_models import Zheng07Cens


fixed_seed = 43


def test1():
    """ Test demonstrates that the baseline <Ncen> is preserved by the
    continuous assembly bias model
    """
    baseline_model = Zheng07Cens()
    with NumpyRNGContext(fixed_seed):
        uran = np.random.rand(10)
    raise NotImplementedError

