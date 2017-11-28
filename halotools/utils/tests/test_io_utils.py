"""
"""
from astropy.utils.data import get_pkg_data_filename

from ..io_utils import file_len

__all__ = ('test_file_len', )


def test_file_len():
    fname = get_pkg_data_filename('data/dummy_ascii.dat')
    assert file_len(fname) == 4
