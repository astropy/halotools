# -*- coding: utf-8 -*-
"""
Methods and classes to read ASCII files storing simulation data. 

"""

__all__ = ('RockstarHlistReader', )

import os
import gzip
from time import time
import numpy as np
from difflib import get_close_matches
from astropy.table import Table

from . import catalog_manager, supported_sims, sim_defaults, cache_config
from ..utils import convert_to_ndarray

from ..custom_exceptions import *

class RockstarHlistReader(object):
    """ Class containing methods used to read raw ASCII data generated with Rockstar. 
    """
    def __init__(self, input_fname, dt, **kwargs):
        """
        Parameters 
        -----------
        input_fname : string 
            Absolute path of the file to be processed. 

        dt : Numpy dtype object 
            The ``dt`` argument instructs the reader how to interpret the 
            columns stored in the ASCII data. 

        column_indices_to_keep : list, optional 
            List of indices of columns that will be stored in the processed catalog. 
            Default behavior is to keep all columns. 

        row_cuts : list, optional 
            List of tuples used to define which rows of the ASCII data will be kept.
            Default behavior is to make no cuts. 

            The row-cut is determined from a list of tuples as follows. 
            Each element of the ``row_cuts`` list is a three-element tuple. 
            The first tuple element will be interpreted as the index of the column 
            upon which your cut is made. 
            The column-indexing convention is C-like, 
            so that the first column has column-index = 0. 
            The second and third tuple elements will be interpreted 
            as lower and upper bounds on this column, respectively. 

            For example, if you only want to keep halos 
            with :math:`M_{\\rm peak} > 1e10`, 
            and :math:`M_{\\rm peak}` is the tenth column of the ASCII file, 
            then you would set row_cuts = [(9, 1e10, float("inf"))]. 

            For any column-index appearing in ``row_cuts``, this index must also 
            appear in ``column_indices_to_keep``: for purposes of good bookeeping, 
            you are not permitted to place a cut on a column that you do not keep. 

        header_char : str, optional
            String to be interpreted as a header line of the ascii hlist file. 
            Default is '#'. 

        Notes 
        -----
        Making even very conservative cuts on 
        either present-day or peak halo mass 
        can result in dramatic reductions in file size; 
        most halos in a typical raw catalog are right on the hairy edge 
        of the numerical resolution limit. 

        Also note that the processed halo catalogs 
        provided by Halotools *do* make cuts on rows. So if you run the 
        `RockstarHlistReader` with default settings on one of the raw 
        catalogs provided by Halotools, you will *not* get a  
        result that matches the corresponding processed catalog 
        provided by Halotools. 

        """

        self._process_constructor_inputs(input_fname, dt, **kwargs)


    def _process_constructor_inputs(self, input_fname, dt, 
        header_char='#', **kwargs):
        """
        """
        self.header_char = header_char

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
        self.fname = input_fname

        self._determine_compression_safe_file_opener()

        self.num_cols_total = self.infer_number_of_columns()

        try:
            assert type(dt) == np.dtype
            assert len(dt) <= self.num_cols_total
        except:
            msg = ("\nInput ``dt`` must be a Numpy dtype object.\n")
            raise HalotoolsError(msg)
        self.dt = dt

        try:
            column_indices_to_keep = kwargs['column_indices_to_keep']
            assert type(column_indices_to_keep) == list
            assert len(column_indices_to_keep) <= self.num_cols_total
            assert len(column_indices_to_keep) == len(self.dt)
            assert set(column_indices_to_keep).issubset(set(xrange(self.num_cols_total)))
        except KeyError:
            column_indices_to_keep = list(xrange(self.num_cols_total))
        except AssertionError:
            msg = ("\nInput ``column_indices_to_keep`` must be a list of integers\n"
                "between zero and and the total number of ascii data columns,\n"
                "and the length of ``column_indices_to_keep`` must equal "
                "the length of the input ``dt``.\n")
            raise HalotoolsError(msg)
        self.column_indices_to_keep = column_indices_to_keep

        try:
            row_cuts = kwargs['row_cuts']
            assert type(row_cuts) == list
            assert len(row_cuts) <= len(self.dt)
            for entry in row_cuts:
                assert type(entry) == tuple
                assert len(entry) == 3
                assert entry[0] in column_indices_to_keep
        except KeyError:
            row_cuts = []
        except AssertionError:
            msg = ("\nInput ``row_cuts`` must be a list of 3-element tuples. \n"
                "The first entry is an integer that will be interpreted as the \n"
                "column-index upon which a cut is made.\n"
                "All column indices must appear in the input ``column_indices_to_keep``.\n"
                )
            raise HalotoolsError(msg)
        self._set_row_cuts(row_cuts)

        try:
            assert (type(header_char) == str) or (type(header_char) == unicode)
            assert len(header_char) == 1
        except AssertionError:
            msg = ("\nThe input ``header_char`` must be a single string character.\n")
            raise HalotoolsError(msg)


    def _determine_compression_safe_file_opener(self):
        """
        """
        try:
            f = gzip.open(self.fname, 'r')
            f.readline()
            self._compression_safe_file_opener = gzip.open
            f.close()
        except IOError:
            self._compression_safe_file_opener = open

    def _set_row_cuts(self, input_row_cuts):
        """
        """
        # Initialize a multi-dimensional array 
        # where we have num_cols_total entries of (-inf, inf)
        x = np.zeros(self.num_cols_total*2) + float("inf")
        x[0::2] = float("-inf")
        x = x.reshape(self.num_cols_total, 2)

        # For any column in x for which there is a cut on the corresponding halo property, 
        # overwrite the column with the two-element tuple defining the cut (lower_bound, upper_bound)
        for entry in input_row_cuts:
            x[entry[0]] = entry[1:]

        self.row_cuts = []
        for column_index in self.column_indices_to_keep:
            self.row_cuts.append((column_index, x[column_index][0], x[column_index][1]))

        # self.row_cuts is now a list of tuples
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
            Nchunks = kwargs['Nchunks']
        except KeyError:
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
        print(" Total number of rows in file = %i" % num_data_rows)
        print(" Number of rows in detected header = %i \n" % header_length)
        if Nchunks==1:
            print("Reading catalog in a single chunk of size %i\n" % chunksize)
        else:
            print("...Reading catalog in %i chunks, each with %i rows\n" % (Nchunks, chunksize))            

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

        return Table(full_array)



