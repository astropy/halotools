"""
"""
import os
import shutil
import numpy as np
from unittest import TestCase
import pytest
from astropy.table import Table

from astropy.config.paths import _find_home

from ..tabular_ascii_reader import TabularAsciiReader


# Determine whether the machine is mine
# This will be used to select tests whose
# returned values depend on the configuration
# of my personal cache directory files
aph_home = '/Users/aphearin'
detected_home = _find_home()
if aph_home == detected_home:
    APH_MACHINE = True
else:
    APH_MACHINE = False

__all__ = ('TestTabularAsciiReader', )


def write_tabular_data(fname):
    with open(fname, 'w') as f:
        f.write('# id  vmax  mvir  upid\n')
        f.write('100  100.  1e9  3999494332\n')
        f.write('101  200.  1e10  -1\n')
        f.write('102  300.  1e11  3999494331\n')
        f.write('103  400.  1e12  3999494332\n')


class TestTabularAsciiReader(TestCase):

    def setUp(self):

        self.tmpdir = os.path.join(_find_home(), '.temp_halotools_testing_dir')
        try:
            os.makedirs(self.tmpdir)
        except OSError:
            pass

        basename = 'abc.txt'
        self.dummy_fname = os.path.join(self.tmpdir, basename)
        _t = Table({'x': [0]})
        _t.write(self.dummy_fname, format='ascii')

    def test_get_fname(self):
        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4')})

        with pytest.raises(IOError) as err:
            reader = TabularAsciiReader(
                os.path.basename(self.dummy_fname),
                columns_to_keep_dict={'mass': (2, 'f4')})
        substr = 'is not a file'
        assert substr in err.value.args[0]

    def test_get_header_char(self):
        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4')},
            header_char='*')

        with pytest.raises(TypeError) as err:
            reader = TabularAsciiReader(self.dummy_fname,
                columns_to_keep_dict={'mass': (2, 'f4')},
                header_char='###')
        substr = 'must be a single string/bytes character'
        assert substr in err.value.args[0]

    def test_process_columns_to_keep(self):

        with pytest.raises(TypeError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4', 'c')},
                header_char='*')
        substr = 'must be a two-element tuple.'
        assert substr in err.value.args[0]

        with pytest.raises(TypeError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict={'mass': (3.5, 'f4')},
                header_char='*')
        substr = 'The first element of the two-element tuple'
        assert substr in err.value.args[0]

        with pytest.raises(TypeError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict={'mass': (2, 'Jose Canseco')},
                header_char='*')
        substr = 'The second element of the two-element tuple'
        assert substr in err.value.args[0]

    def test_verify_input_row_cuts(self):

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4')},
            row_cut_min_dict={'mass': 8})

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4')},
            row_cut_max_dict={'mass': 8})

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4')},
            row_cut_eq_dict={'mass': 8})

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4')},
            row_cut_neq_dict={'mass': 8})

    def test_verify_min_max_consistency(self):

        reader = TabularAsciiReader(
            self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4')},
            row_cut_min_dict={'mass': 8}, row_cut_max_dict={'mass': 9})

        with pytest.raises(ValueError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4')},
                row_cut_min_dict={'mass': 9}, row_cut_max_dict={'mass': 8})
        substr = 'This will result in zero selected rows '
        assert substr in err.value.args[0]

        with pytest.raises(KeyError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict={'mass': (2, 'f4')},
                row_cut_min_dict={'mass': 9}, row_cut_max_dict={'vmax': 8})
        substr = 'The ``vmax`` key does not appear in the input'
        assert substr in err.value.args[0]

        with pytest.raises(KeyError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict={'vmax': (1, 'f4')},
                row_cut_min_dict={'mass': 9}, row_cut_max_dict={'vmax': 8})
        substr = 'The ``mass`` key does not appear in the input'
        assert substr in err.value.args[0]

    def test_verify_eq_neq_consistency(self):

        with pytest.raises(ValueError) as err:
            reader = TabularAsciiReader(
                self.dummy_fname, columns_to_keep_dict={'mvir': (2, 'f4')},
                row_cut_eq_dict={'mvir': 8}, row_cut_neq_dict={'mvir': 8})
        substr = 'This will result in zero selected rows '
        assert substr in err.value.args[0]

    def test_read_dummy_halo_catalog1(self):
        fname = 'abc'
        columns_to_keep_dict = {'vmax': (1, 'f4')}

        with pytest.raises(IOError) as err:
            reader = TabularAsciiReader(fname, columns_to_keep_dict)
        substr = "is not a file"
        assert substr in err.value.args[0]

    def test_read_dummy_halo_catalog2(self):
        fname = self.dummy_fname
        write_tabular_data(fname)
        columns_to_keep_dict = {'upid': (3, 'i8'), 'vmax': (3, 'f4')}

        with pytest.raises(ValueError) as err:
            reader = TabularAsciiReader(fname, columns_to_keep_dict)
        substr = "appears more than once in your ``columns_to_keep_dict``"
        assert substr in err.value.args[0]

    @pytest.mark.slow
    def test_read_dummy_halo_catalog3(self):
        columns_to_keep_dict = {'vmax': (1, 'f4')}
        write_tabular_data(self.dummy_fname)
        row_cut_min_dict = {'vmax': 0.5}
        row_cut_max_dict = {'vmax': 0.4}

        with pytest.raises(ValueError) as err:
            reader = TabularAsciiReader(self.dummy_fname, columns_to_keep_dict,
                row_cut_min_dict=row_cut_min_dict, row_cut_max_dict=row_cut_max_dict)
        substr = "This will result in zero selected rows and is not permissible."
        assert substr in err.value.args[0]

    @pytest.mark.slow
    def test_read_dummy_halo_catalog4(self):
        columns_to_keep_dict = {'vmax': (1, 'f4')}
        write_tabular_data(self.dummy_fname)

        row_cut_eq_dict = {'vmax': 0.5}
        row_cut_neq_dict = {'vmax': 0.5}

        with pytest.raises(ValueError) as err:
            reader = TabularAsciiReader(self.dummy_fname, columns_to_keep_dict,
                row_cut_eq_dict=row_cut_eq_dict, row_cut_neq_dict=row_cut_neq_dict)
        substr = "This will result in zero selected rows and is not permissible."
        assert substr in err.value.args[0]

    @pytest.mark.slow
    def test_read_dummy_halo_catalog5(self):
        """
        """
        write_tabular_data(self.dummy_fname)

        columns_to_keep_dict = {'vmax': (1, 'f4'), 'id': (0, 'i8'), 'upid': (3, 'i8')}

        reader = TabularAsciiReader(self.dummy_fname, columns_to_keep_dict,
            row_cut_min_dict={'vmax': 101},
            row_cut_max_dict={'vmax': 399},
            row_cut_eq_dict={'upid': -1},
            row_cut_neq_dict={'id': -1}
            )

        arr = reader.read_ascii()

        # Verify that the cuts were applied correctly
        assert np.all(arr['vmax'] >= 101)
        assert np.all(arr['vmax'] <= 399)
        assert np.all(arr['upid'] == -1)

        # verify that the cuts were non-trivial
        reader = TabularAsciiReader(self.dummy_fname, columns_to_keep_dict)
        arr = reader.read_ascii()
        assert np.any(arr['vmax'] < 101)
        assert np.any(arr['vmax'] > 399)
        assert np.any(arr['upid'] != -1)

    @pytest.mark.slow
    def test_read_dummy_halo_catalog6(self):
        """
        """
        write_tabular_data(self.dummy_fname)

        columns_to_keep_dict = {'vmax': (1, 'f4'), 'id': (0, 'i8'), 'upid': (3, 'i8')}

        reader = TabularAsciiReader(self.dummy_fname, columns_to_keep_dict)

        with pytest.raises(ValueError) as err:
            arr = reader.read_ascii(chunk_memory_size=0)
        substr = "Must choose non-zero size for input ``chunk_memory_size``"
        assert substr in err.value.args[0]

    def tearDown(self):
        try:
            shutil.rmtree(self.tmpdir)
        except:
            pass
