# -*- coding: utf-8 -*-
"""
Module storing the TabularAsciiReader, a class providing a memory-efficient 
algorithm for reading a very large ascii file that stores tabular data 
of a data type that is known in advance. 

"""

__all__ = ('TabularAsciiReader', )

import os
import gzip
from time import time
import numpy as np

from ..custom_exceptions import HalotoolsError

class TabularAsciiReader(object):
    """ Class providing a memory-efficient algorithm for 
    reading a very large ascii file that stores tabular data 
    of a data type that is known in advance. 

    When reading ASCII data with 
    `~halotools.sim_manager.TabularAsciiReader.read_ascii`, user-defined 
    cuts on columns are applied on-the-fly using a python generator 
    to yield only those columns whose indices appear in the 
    input ``columns_to_keep_dict``. 

    The data is generated in chunks, 
    and a user-defined mask is applied to each chunk. 
    The only aggregated data from each chunk are those rows  
    passing all requested cuts, so that the 
    `~halotools.sim_manager.TabularAsciiReader` 
    only requires you to have enough RAM to store the *cut* catalog, 
    not the entire ASCII file. 

    The primary method is the class is  
    `~halotools.sim_manager.TabularAsciiReader.read_ascii`. 
    The output of this method is a structured Numpy array, 
    which can then be stored in your preferred binary format 
    using the built-in Numpy methods, h5py, etc. 

    The algorithm assumes that data of known, unchanging type is 
    arranged in a consecutive sequence of lines within the ascii file, 
    and that the appearance of an empty line demarcates the end 
    of the data stream. 
    """
    def __init__(self, input_fname, columns_to_keep_dict, 
        header_char='#', row_cut_min_dict = {}, row_cut_max_dict = {}, 
        row_cut_eq_dict = {}, row_cut_neq_dict = {}):
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

            For example, if row_cut_eq_dict = {'upid': -1}, then *all* rows of the 
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

            For example, if row_cut_neq_dict = {'upid': -1}, then *no* rows of the 
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

        self._process_columns_to_keep(columns_to_keep_dict)

        self.row_cut_min_dict = row_cut_min_dict
        self.row_cut_max_dict = row_cut_max_dict
        self.row_cut_eq_dict = row_cut_eq_dict
        self.row_cut_neq_dict = row_cut_neq_dict

        self._verify_input_row_cuts_keys()
        self._verify_min_max_consistency()
        self._verify_eq_neq_consistency()

    def _verify_input_row_cuts_keys(self, **kwargs):
        """ Require all columns upon which a row-cut is placed to also appear in 
        the input ``columns_to_keep_dict``. For purposes of good bookeeping, 
        you are not permitted to place a cut on a column that you do not keep.        
        """
        potential_row_cuts = ('row_cut_min_dict', 'row_cut_max_dict', 
            'row_cut_eq_dict', 'row_cut_neq_dict')
        for row_cut_key in potential_row_cuts:

            row_cut_dict = getattr(self, row_cut_key)

            for key in row_cut_dict:
                try:
                    assert key in self.input_columns_to_keep_dict.keys()
                except AssertionError:
                    msg = ("\nThe ``"+key+"`` key does not appear in the input \n"
                        "``columns_to_keep_dict``, but it does appear in the "
                        "input ``"+row_cut_key+"``. \n"
                        "It is not permissible to place a cut "
                        "on a column that you do not keep.\n")
                    raise HalotoolsError(msg)

    def _verify_min_max_consistency(self, **kwargs):
        """ Verify that no min_cut column has a value greater to the corresponding max_cut. 

        Such a choice would laboriously result in a final catalog with zero entries. 
        """

        for row_cut_min_key, row_cut_min in self.row_cut_min_dict.iteritems():
            try:
                row_cut_max = self.row_cut_max_dict[row_cut_min_key]
                if row_cut_max <= row_cut_min:
                    msg = ("\nFor the ``"+row_cut_min_key+"`` column, \n"
                        "you set the value of the input ``row_cut_min_dict`` to "
                        +str(row_cut_min)+"\nand the value of the input "
                        "``row_cut_max_dict`` to "+str(row_cut_max)+"\n"
                        "This will result in zero selected rows and is not permissible.\n")
                    raise HalotoolsError(msg)
            except KeyError:
                pass

        for row_cut_max_key, row_cut_max in self.row_cut_max_dict.iteritems():
            try:
                row_cut_min = self.row_cut_min_dict[row_cut_max_key]
                if row_cut_min >= row_cut_max:
                    msg = ("\nFor the ``"+row_cut_max_key+"`` column, \n"
                        "you set the value of the input ``row_cut_max_dict`` to "
                        +str(row_cut_max)+"\nand the value of the input "
                        "``row_cut_min_dict`` to "+str(row_cut_min)+"\n"
                        "This will result in zero selected rows and is not permissible.\n")
                    raise HalotoolsError(msg)
            except KeyError:
                pass


    def _verify_eq_neq_consistency(self, **kwargs):
        """ Verify that no neq_cut column has a value equal to the corresponding eq_cut. 

        Such a choice would laboriously result in a final catalog with zero entries. 
        """ 

        for row_cut_eq_key, row_cut_eq in self.row_cut_eq_dict.iteritems():
            try:
                row_cut_neq = self.row_cut_neq_dict[row_cut_eq_key]
                if row_cut_neq == row_cut_eq:
                    msg = ("\nFor the ``"+row_cut_eq_key+"`` column, \n"
                        "you set the value of the input ``row_cut_eq_dict`` to "
                        +str(row_cut_eq)+"\nand the value of the input "
                        "``row_cut_neq_dict`` to "+str(row_cut_neq)+"\n"
                        "This will result in zero selected rows and is not permissible.\n")
                    raise HalotoolsError(msg)
            except KeyError:
                pass

        for row_cut_neq_key, row_cut_neq in self.row_cut_neq_dict.iteritems():
            try:
                row_cut_eq = self.row_cut_eq_dict[row_cut_neq_key]
                if row_cut_eq == row_cut_neq:
                    msg = ("\nFor the ``"+row_cut_neq_key+"`` column, \n"
                        "you set the value of the input ``row_cut_neq_dict`` to "
                        +str(row_cut_neq)+"\nand the value of the input "
                        "``row_cut_eq_dict`` to "+str(row_cut_eq)+"\n"
                        "This will result in zero selected rows and is not permissible.\n")
                    raise HalotoolsError(msg)
            except KeyError:
                pass

    def _process_columns_to_keep(self, columns_to_keep_dict):
        """ Private method performs sanity checks in the input ``columns_to_keep_dict`` 
        and uses this input to define two attributes used for future bookkeeping, 
        ``self.column_indices_to_keep`` and ``self.dt``. 
        """

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


        self.column_indices_to_keep = list(
            [columns_to_keep_dict[key][0] for key in columns_to_keep_dict])

        self.dt = np.dtype(
            [(key, columns_to_keep_dict[key][1]) 
            for key in columns_to_keep_dict]
            )

        self.input_columns_to_keep_dict = columns_to_keep_dict

    def _get_fname(self, input_fname):
        """ Verify that the input fname does not already exist. 
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
        """ Verify that the input header_char is 
        a one-character string or unicode variable. 

        """
        try:
            assert (type(header_char) == str) or (type(header_char) == unicode)
            assert len(header_char) == 1
        except AssertionError:
            msg = ("\nThe input ``header_char`` must be a single string character.\n")
            raise HalotoolsError(msg)
        return header_char

    def _determine_compression_safe_file_opener(self):
        """ Determine whether to use *open* or *gzip.open* to read 
        the input file, depending on whether or not it is compressed. 
        """
        f = gzip.open(self.fname, 'r')
        try:
            f.read(1)
            self._compression_safe_file_opener = gzip.open
        except IOError:
            self._compression_safe_file_opener = open
        finally:
            f.close()

    def header_len(self):
        """ Number of rows in the ascii data header. 

        Parameters 
        ----------
        fname : string 

        Returns 
        -------
        Nheader : int

        Notes 
        -----
        The header is assumed to be those characters at the beginning of the file 
        that begin with ``self.header_char``. 

        All empty lines that appear in header will be included in the count. 

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
        """ 
        Number of rows data in the input ASCII file. 

        Returns 
        --------
        Nrows_data : int 
            Total number of rows of data. 

        Notes 
        -------
        The returned value is computed as the number of lines 
        between the returned value of `header_len` and 
        the next appearance of "\n" as the sole character on a line. 

        The `data_len` method is the particular section of code 
        where where the following assumptions are made:

        1. The data begins with the first appearance of a line that does not begin with ``self.header_char``. 

        2. The data ends with the first subsequent appearance of an empty line. 
        """
        Nrows_data = 0
        with self._compression_safe_file_opener(self.fname, 'r') as f:
            for i, l in enumerate(f):
                if ( (l[0:len(self.header_char)]!=self.header_char) and (l!="\n") ):
                    Nrows_data += 1
        return Nrows_data

    def data_chunk_generator(self, chunk_size, f):
        """
        Python generator uses *f.readline()* to march 
        through an input open file object to yield 
        a chunk of data of the input size. 
        The generator only yields columns 
        that were included in the ``columns_to_keep_dict`` 
        that was passed to the constructor. 

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
        """ Method applies a boolean mask to the input array 
        based on the row-cuts determined by the 
        dictionaries passed to the constructor. 

        Parameters 
        -----------
        array_chunk : Numpy array  

        Returns 
        --------
        cut_array : Numpy array             
        """ 
        mask = np.ones(len(array_chunk), dtype = bool)

        for colname, lower_bound in self.row_cut_min_dict.iteritems():
            mask *= array_chunk[colname] > lower_bound

        for colname, upper_bound in self.row_cut_max_dict.iteritems():
            mask *= array_chunk[colname] < upper_bound

        for colname, equality_condition in self.row_cut_eq_dict.iteritems():
            mask *= array_chunk[colname] == equality_condition

        for colname, inequality_condition in self.row_cut_neq_dict.iteritems():
            mask *= array_chunk[colname] != inequality_condition

        return array_chunk[mask]

    def read_ascii(self, chunk_memory_size = 500):
        """ Method reads the input ascii and returns 
        a structured Numpy array of the data 
        that passes the row- and column-cuts. 

        Parameters 
        ----------
        chunk_memory_size : int, optional 
            Determine the approximate amount of Megabytes of memory 
            that will be processed in chunks. This variable 
            must be smaller than the amount of RAM on your machine; 
            choosing larger values typically improves performance. 
            Default is 500 Mb. 

        Returns 
        --------
        full_array : array_like 
            Structured Numpy array storing the rows and columns 
            that pass the input cuts. The columns of this array 
            are those selected by the ``column_indices_to_keep`` 
            argument passed to the constructor. 

        See also 
        ----------
        data_chunk_generator
        """
        start = time()

        file_size = os.path.getsize(fname) 
        num_data_rows = self.data_len()
        # Set the number of chunks to be filesize/chunk_memory, 
        # but enforcing that 0 < Nchunks <= num_data_rows
        try:
            Nchunks = max(1, min(file_size / chunk_memory_size, num_data_rows))
        except ZeroDivisionError:
            msg = ("\nMust choose non-zero size for input ``chunk_memory_size``")
            raise HalotoolsError(msg)

        num_rows_in_chunk = int(num_data_rows / float(Nchunks))
        num_full_chunks = num_data_rows / num_rows_in_chunk
        num_rows_in_chunk_remainder = num_data_rows - num_rows_in_chunk*Nchunks

        print("\n...Processing ASCII data of file: \n%s\n " % self.fname)
        print(" Total number of rows in detected data = %i" % num_data_rows)
        print(" Number of rows in detected header = %i \n" % header_length)

        header_length = self.header_len()
        chunklist = []
        with self._compression_safe_file_opener(self.fname, 'r') as f:

            for skip_header_row in xrange(header_length):
                _ = f.readline()

            for ichunk in xrange(num_full_chunks):

                chunk_array = np.array(list(
                    self.data_chunk_generator(num_rows_in_chunk, f)), dtype=self.dt)
                cut_chunk = self.apply_row_cut(chunk_array)
                chunklist.append(cut_chunk)

            # Now for the remainder chunk
            chunk_array = np.array(list(
                self.data_chunk_generator(num_rows_in_chunk_remainder, f)), dtype=self.dt)
            cut_chunk = self.apply_row_cut(chunk_array)
            chunklist.append(cut_chunk)

        full_array = np.concatenate(chunklist)
                
        end = time()
        runtime = (end-start)

        if runtime > 60:
            runtime = runtime/60.
            msg = "Total runtime to read in ASCII = %.1f minutes\n"
        else:
            msg = "Total runtime to read in ASCII = %.2f seconds\n"
        print(msg % runtime)
        print("\a")

        return full_array



