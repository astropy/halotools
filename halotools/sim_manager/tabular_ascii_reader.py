# -*- coding: utf-8 -*-
"""
Methods and classes to read ASCII files storing simulation data. 

"""

__all__ = ('TabularAsciiReader', )

import os
import gzip
from time import time
import numpy as np
from astropy.table import Table
from copy import deepcopy

from ..utils import convert_to_ndarray
from ..custom_exceptions import HalotoolsError

class TabularAsciiReader(object):
    """ Class containing methods used to read raw ASCII data generated with Rockstar. 
    """
    def __init__(self, input_fname, columns_to_keep_dict, 
        header_char='#', **kwargs):
        """
        Parameters 
        -----------
        input_fname : string 
            Absolute path of the file to be processed. 

        columns_to_keep_dict : dict 
            Dictionary used to define which columns 
            of the tabular ASCII data will be kept.

            Each key of the dictionary will be the name of the 
            column in the returned data table. The value bound to 
            each key is a two-element tuple. 
            The first tuple entry is an integer providing 
            the *index* of the column to be kept, starting from 0. 
            The second tuple entry is a string defining the Numpy dtype 
            of the data in that column, 
            e.g., 'f4' for a float, 'f8' for a double, 
            or 'i8' for a long. 

            Thus an example ``columns_to_keep_dict`` could be 
            {'mass': (1, 'f4'), 'obj_id': (0, 'i8'), 'spin': (45, 'f4')}

        row_cut_min_dict : dict, optional 
            Dictionary used to place a lower-bound cut on the rows 
            of the tabular ASCII data. 

            Each key of the dictionary must also 
            be a key of the input ``columns_to_keep_dict``; 
            for purposes of good bookeeping, you are not permitted 
            to place a cut on a column that you do not keep. The value 
            bound to each key serves as the lower bound on the data stored 
            in that row. A row with a smaller value than this lower bound for the 
            corresponding column will not appear in the returned data table. 

            For example, if row_cut_min_dict = {'mass': 1e10}, then all rows of the 
            returned data table will have a mass greater than 1e10. 

        row_cut_max_dict : dict, optional 
            Dictionary used to place an upper-bound cut on the rows 
            of the tabular ASCII data. 

            Each key of the dictionary must also 
            be a key of the input ``columns_to_keep_dict``; 
            for purposes of good bookeeping, you are not permitted 
            to place a cut on a column that you do not keep. The value 
            bound to each key serves as the upper bound on the data stored 
            in that row. A row with a larger value than this upper bound for the 
            corresponding column will not appear in the returned data table. 

            For example, if row_cut_min_dict = {'mass': 1e15}, then all rows of the 
            returned data table will have a mass less than 1e15. 

        row_cut_eq_dict : dict, optional 
            Dictionary used to place an equality cut on the rows 
            of the tabular ASCII data. 

            Each key of the dictionary must also 
            be a key of the input ``columns_to_keep_dict``; 
            for purposes of good bookeeping, you are not permitted 
            to place a cut on a column that you do not keep. The value 
            bound to each key serves as the required value for the data stored 
            in that row. Only rows with a value equal to this required value for the 
            corresponding column will appear in the returned data table. 

            For example, if row_cut_eq_dict = {'upid': -1}, then all rows of the 
            returned data table will have a upid of -1. 

        row_cut_neq_dict : dict, optional 
            Dictionary used to place an inequality cut on the rows 
            of the tabular ASCII data. 

            Each key of the dictionary must also 
            be a key of the input ``columns_to_keep_dict``; 
            for purposes of good bookeeping, you are not permitted 
            to place a cut on a column that you do not keep. The value 
            bound to each key serves as a forbidden value for the data stored 
            in that row. Rows with a value equal to this forbidden value for the 
            corresponding column will not appear in the returned data table. 

            For example, if row_cut_neq_dict = {'upid': -1}, then no rows of the 
            returned data table will have a upid of -1. 

        header_char : str, optional
            String to be interpreted as a header line of the ascii hlist file. 
            Default is '#'. 

        Notes 
        ------
        When the ``row_cut_min_dict``, ``row_cut_max_dict``, 
        ``row_cut_eq_dict`` and ``row_cut_neq_dict`` keyword arguments are used 
        simultaneously, only rows passing all cuts will be kept. 

        """

        self.fname = self._get_fname(input_fname)

        self.header_char = self._get_header_char(header_char)

        self._determine_compression_safe_file_opener()

        self.num_cols_total = self.infer_number_of_columns()

        self._process_columns_to_keep(columns_to_keep_dict)

        # input_row_cuts = self._interpret_input_row_cuts(**kwargs)
        # self.row_cuts = self._get_row_cuts(input_row_cuts)

    def _verify_input_row_cuts(self, **kwargs):
        """
        """
        potential_row_cuts = ('row_cut_min_dict', 'row_cut_max_dict', 
            'row_cut_eq_dict', 'row_cut_neq_dict')
        for row_cut_key in potential_row_cuts:

            try:
                row_cut_type = kwargs[row_cut_key]

                for key in row_cut_dict:
                    try:
                        assert key in self.input_columns_to_keep_dict.keys()
                    except AssertionError:
                        msg = ("\nThe ``"+key+"`` key does not appear in the input \n"
                            "``columns_to_keep_dict``, but it does appear in the "
                            "input ``"+row_cut_type+"``. \n"
                            "It is not permissible to place a cut "
                            "on a column that you do not keep.\n")
                        raise HalotoolsError(msg)
            except KeyError:
                pass

        try:
            row_cut_min_dict = kwargs['row_cut_min_dict']

            try:
                row_cut_max_dict = kwargs['row_cut_max_dict']
                for row_cut_min_key, row_cut_min in row_cut_min_dict.iteritems():
                    try:
                        row_cut_max = row_cut_max_dict[row_cut_min_key]
                        if row_cut_max <= row_cut_min:
                            msg = ("\nFor the ``"+row_cut_min_key+"`` column, \n"
                                "you set the value of the input ``row_cut_min_dict`` to "
                                +str(row_cut_min)+"\nand the value of the input "
                                "``row_cut_max_dict`` to "+str(row_cut_max)+"\n"
                                "This will result in zero selected rows and is not permissible.\n")
                            raise HalotoolsError(msg)
                    except KeyError:
                        pass

            except KeyError:
                pass

        except KeyError:
            pass


    def _process_columns_to_keep(self, columns_to_keep_dict):

        for key, value in columns_to_keep_dict.iteritems():
            try:
                assert type(value) == tuple
                assert len(value) == 2
            except AssertionError:
                msg = ("\nThe value bound to every key of the input ``columns_to_keep_dict``\n"
                    "must be a two-element tuple.\n"
                    "The ``"+key+"`` is not the required type.\n"
                    )
                raise HalotoolsError(msg)

            column_index, dtype = value
            try:
                assert type(column_index) == int
            except AssertionError:
                msg = ("\nThe first element of the two-element tuple bound to every key of \n"
                    "the input ``columns_to_keep_dict`` must an integer.\n"
                    "The first element of the ``"+key+"`` is not the required type.\n"
                    )
                raise HalotoolsError(msg)
            try:
                dt = np.dtype(dtype)
            except:
                msg = ("\nThe second element of the two-element tuple bound to every key of \n"
                    "the input ``columns_to_keep_dict`` must be a string recognized by Numpy\n"
                    "as a data type, e.g., 'f4' or 'i8'.\n"
                    "The second element of the ``"+key+"`` is not the required type.\n"
                    )
                raise HalotoolsError(msg)

            attr_name = key + '_column_index'
            setattr(self, attr_name, value[0])
            attr_name = key + '_dtype'
            setattr(self, attr_name, np.dtype(value[1]))

        self.dt = np.dtype(
            [(key, columns_to_keep_dict[key][1]) 
            for key in columns_to_keep_dict]
            )

        self.input_columns_to_keep_dict = columns_to_keep_dict

    def _get_fname(self, input_fname):
        """
        """
        # Check whether input_fname exists. 
        if not os.path.isfile(input_fname):
            # Check to see whether the uncompressed version is available instead
            if not os.path.isfile(input_fname[:-3]):
                msg = "Input filename %s is not a file" 
                raise HalotoolsError(msg % input_fname)
            else:
                msg = ("Input filename ``%s`` is not a file. \n"
                    "However, ``%s`` exists, so change your input_fname accordingly.")
                raise HalotoolsError(msg % (input_fname, input_fname[:-3]))

        return input_fname

    def _get_header_char(self, header_char):
        """
        """
        try:
            assert (type(header_char) == str) or (type(header_char) == unicode)
            assert len(header_char) == 1
        except AssertionError:
            msg = ("\nThe input ``header_char`` must be a single string character.\n")
            raise HalotoolsError(msg)
        return header_char

    def _determine_columns_to_keep(self, **kwargs):
        """
        """
        try:
            columns_to_keep = kwargs['columns_to_keep']
        except KeyError:
            columns_to_keep = self._infer_columns_to_keep_from_ascii(**kwargs)
        self._test_columns_to_keep(columns_to_keep)
        
        return columns_to_keep

    def _test_columns_to_keep(self, columns_to_keep):
        """
        """
        for entry in columns_to_keep:
            try:
                assert len(entry) == 3
                assert type(entry[0]) == int
                assert (type(entry[1]) == str) or (type(entry[1]) == unicode)
                assert (type(entry[2]) == str) or (type(entry[2]) == unicode)
                dt = np.dtype([(entry[1], entry[2])])
            except:
                msg = ("\nYour input ``columns_to_keep`` is not properly formatted.\n"
                    "See the docstring of `RockstarHlistReader` for a description.\n")
                raise HalotoolsError(msg)
            try:
                assert entry[0] < self.num_cols_total
            except:
                msg = ("\nYour ``"+entry[1]+"`` entry of ``columns_to_keep``\n"
                    "has its first tuple element = " + str(entry[0]) + "\n"
                    "But the total number of columns in the hlist file is " + 
                    str(self.num_cols_total) + "\nRemember that the first column has index 0.\n"
                    )
                raise HalotoolsError(msg)


    def _infer_columns_to_keep_from_ascii(self, **kwargs):
        """

        """
        try:
            columns_fname = kwargs['columns_to_keep_ascii_fname']
            assert os.path.isfile(columns_fname)
            t = Table.read(columns_fname, format='ascii', 
                names = ['column_index', 'halo_property', 'dtype'])
            print("Branch AAA triggered")
            columns_to_keep = [(t['column_index'][i], t['halo_property'][i], 
                t['dtype'][i]) for i in xrange(len(t))]
            return columns_to_keep
        except AssertionError:
            msg = ("\nThe input ``columns_to_keep_ascii_fname`` does not exist.\n"
                "You must specify an absolute path to an existing file.\n")
            raise HalotoolsError(msg)
        except KeyError:
            msg = ("\nIf you do not pass an input ``columns_to_keep`` argument, \n"
                "you must pass an input ``columns_to_keep_ascii_fname`` argument.\n")
            raise HalotoolsError(msg)
        except InconsistentTableError:
            msg = ("\nThe file stored at \n" + columns_fname + "\n"
                "is not properly formatted. See the following file for an example:\n\n"
                "halotools/data/RockstarHlistReader_input_example.dat\n\n")
            raise HalotoolsError(msg)


    def _interpret_column_indices_to_keep(self):
        """
        """
        self.column_indices_to_keep = [entry[0] for entry in self.columns_to_keep]


    def _interpret_input_dt(self):
        """
        """
        dt_list = [(entry[1], entry[2]) for entry in self.columns_to_keep]
        self.dt = np.dtype(dt_list)

    def _determine_compression_safe_file_opener(self):
        """
        """
        f = gzip.open(self.fname, 'r')
        try:
            f.read(1)
            self._compression_safe_file_opener = gzip.open
        except IOError:
            self._compression_safe_file_opener = open
        finally:
            f.close()

    def _interpret_input_row_cuts(self, **kwargs):
        """
        """
        try:
            input_row_cuts = kwargs['row_cuts']
            assert type(input_row_cuts) == list
            assert len(input_row_cuts) <= len(self.dt)
            for entry in input_row_cuts:
                assert type(entry) == tuple
                assert len(entry) == 3
                assert entry[0] in self.dt.names
        except KeyError:
            input_row_cuts = []
        except AssertionError:
            msg = ("\nInput ``row_cuts`` must be a list of 3-element tuples. \n"
                "The first entry must be a string that will be interpreted as the \n"
                "name of the column upon which an on-the-fly cut is made.\n"
                "All such columns must appear in the input ``columns_to_keep``.\n"
                )
            raise HalotoolsError(msg)

        return input_row_cuts

    def _get_row_cuts(self, input_row_cuts):
        """
        """

        # The first entry in each row_cut tuple 
        # contains the string name of the column 
        # Use the columns_to_keep list to replace this entry with the column index
        reformatted_row_cuts = list(
            [deepcopy(a[0]), deepcopy(a[1]), deepcopy(a[2])] 
            for a in input_row_cuts)

        for ii, row_cut in enumerate(input_row_cuts):
            name = row_cut[0]
            # Look for the corresponding entry
            index = np.nan
            for column in self.columns_to_keep:
                if column[1] == name:
                    index = column[0]
            if index == np.nan:
                msg = ("\nYou have made a cut on the ``"+name+"`` halo property\n"
                    "without including this property with the ``columns_to_keep``. "
                    "For bookeeping reasons, this is not permissible.\n"
                    "Either change your cut or include the column.\n")
                raise HalotoolsError(msg)
            else:
                reformatted_row_cuts[ii][0] = index


        # Initialize a multi-dimensional array 
        # where we have num_cols_total entries of (-inf, inf)
        x = np.zeros(self.num_cols_total*2) + float("inf")
        x[0::2] = float("-inf")
        x = x.reshape(self.num_cols_total, 2)

        # For any column in x for which there is a cut on the corresponding halo property, 
        # overwrite the column with the two-element tuple defining the cut (lower_bound, upper_bound)
        for entry in reformatted_row_cuts:
            print(entry)
            x[entry[0]] = entry[1:]

        output_row_cuts = []
        for column_index in self.column_indices_to_keep:
            output_row_cuts.append((column_index, x[column_index][0], x[column_index][1]))

        return output_row_cuts
        # output_row_cuts is now a list of tuples
        # This list has the same number of entries as the number of columns to keep 
        # Each element of this list is a 3-element tuple of the form: 
        # (ascii_column_index, lower_bound, upper_bound)

    def header_len(self):
        """ Compute the number of header rows in the raw halo catalog. 

        Parameters 
        ----------
        fname : string 

        Returns 
        -------
        Nheader : int

        Notes 
        -----
        All empty lines that appear in header 
        will be included in the count. 

        """
        Nheader = 0
        with self._compression_safe_file_opener(self.fname, 'r') as f:
            for i, l in enumerate(f):
                if ( (l[0:len(self.header_char)]==self.header_char) or (l=="\n") ):
                    Nheader += 1
                else:
                    break

        return Nheader

    def data_len(self):
        Nrows_data = 0
        with self._compression_safe_file_opener(self.fname, 'r') as f:
            for i, l in enumerate(f):
                if ( (l[0:len(self.header_char)]!=self.header_char) and (l!="\n") ):
                    Nrows_data += 1
        return Nrows_data

    def infer_number_of_columns(self):
        """ Find the first line of data and infer the total number of columns
        """
        with self._compression_safe_file_opener(self.fname, 'r') as f:
            for i, l in enumerate(f):
                if ( (l[0:len(self.header_char)]!=self.header_char) and (l!="\n") ):
                    line = l.strip().split()
                    return len(line)

    def data_chunk_generator(self, chunk_size, f):
        """
        Parameters 
        -----------
        chunk_size : int 
            Number of rows of data in the chunk being generated 

        f : File
            Open file object being read

        Returns 
        --------
        chunk : tuple 
            Tuple of data from the ascii. 
            Only data from ``column_indices_to_keep`` are yielded. 

        """
        cur = 0
        while cur < chunk_size:
            line = f.readline()    
            parsed_line = line.strip().split()
            yield tuple(parsed_line[i] for i in self.column_indices_to_keep)
            cur += 1 

    def apply_row_cut(self, array_chunk):
        """
        """
        mask = np.ones(len(array_chunk), dtype = bool)
        for idx, entry in enumerate(self.row_cuts):
            mask *= ( 
                (array_chunk[array_chunk.dtype.names[idx]] >= entry[1]) & 
                (array_chunk[array_chunk.dtype.names[idx]] <= entry[2])
                )
        return array_chunk[mask]


    def read_halocat(self, **kwargs):
        """ Reads the raw halo catalog in chunks and returns a structured array
        after applying cuts.

        Parameters 
        ----------
        Nchunks : int, optional 
            `read_halocat` reads and processes ascii 
            in chunks at a time, both to improve performance and 
            so that the entire raw halo catalog need not fit in memory 
            in order to process it. The total number of chunks to use 
            can be specified with the `Nchunks` argument. Default is 1000. 

        """
        start = time()

        try:
            Nchunks = int(kwargs['Nchunks'])
        except:
            Nchunks = 100

        header_length = self.header_len()
        num_data_rows = self.data_len()

        chunksize = int(num_data_rows / float(Nchunks))
        num_full_chunks = num_data_rows/chunksize
        chunksize_remainder = num_data_rows % chunksize
        if chunksize == 0:
            chunksize = num_data_rows # data will not be chunked
            Nchunks = 1

        print("\n...Processing ASCII data of file: \n%s\n " % self.fname)
        print(" Total number of rows containing halo catalog data = %i" % num_data_rows)
        print(" Number of rows in detected header = %i \n" % header_length)

        with self._compression_safe_file_opener(self.fname, 'r') as f:

            for skip_header_row in xrange(header_length):
                _ = f.readline()

            for ichunk in xrange(num_full_chunks):

                chunk_array = np.array(list(
                    self.data_chunk_generator(chunksize, f)), dtype=self.dt)
                cut_chunk = self.apply_row_cut(chunk_array)

                try:
                    # append the new chunk onto the existing array
                    full_array = np.append(full_array, cut_chunk)
                except NameError:
                    # we have just gotten the first chunk
                    full_array = cut_chunk

            # Now for the final chunk
            chunk_array = np.array(list(
                self.data_chunk_generator(chunksize_remainder, f)), dtype=self.dt)
            cut_chunk = self.apply_row_cut(chunk_array)

            try:
                # append the new chunk onto the existing array
                full_array = np.append(full_array, cut_chunk)
            except NameError:
                # There were zero full chunks and so we only have the remainder
                full_array = cut_chunk
                

        end = time()
        runtime = (end-start)
        if runtime > 60:
            runtime = runtime/60.
            msg = "Total runtime to read in ASCII = %.1f minutes\n"
        else:
            msg = "Total runtime to read in ASCII = %.1f seconds\n"
        print(msg % runtime)
        print("\a")

        return Table(full_array)



