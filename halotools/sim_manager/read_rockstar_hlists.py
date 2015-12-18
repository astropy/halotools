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
from copy import deepcopy

from . import catalog_manager, supported_sims, sim_defaults, cache_config
from ..utils import convert_to_ndarray

from ..custom_exceptions import *

class RockstarHlistReader(object):
    """ Class containing methods used to read raw ASCII data generated with Rockstar. 
    """
    def __init__(self, input_fname, header_char='#', **kwargs):
        """
        Parameters 
        -----------
        input_fname : string 
            Absolute path of the file to be processed. 

        columns_to_keep : list, optional 
            List of tuples used to define which columns 
            of the halo catalog ASCII data will be kept.
            If ``columns_to_keep`` is not specified, the 
            ``columns_to_keep_ascii_fname`` keyword must be provided. 

            For every desired column of data, ``columns_to_keep`` 
            should have a corresponding list element. 
            Each list element is a tuple with 3 entries. 
            The first tuple entry is an integer providing 
            the *index* of the column to be kept, starting from 0. 
            The second tuple entry is a string providing the name 
            that will be given to the data in that column, 
            e.g., 'halo_id' or 'halo_spin'. 
            The third tuple entry is a string defining the Numpy dtype 
            of the data in that column, 
            e.g., 'f4' for a float, 'f8' for a double, 
            or 'i8' for a long. 

            Thus an example input for ``columns_to_keep`` could be 
            [(0, 'halo_scale_factor', 'f4'), (1, 'halo_id', 'i8'), (16, 'halo_vmax', 'f4')]. 

        columns_to_keep_ascii_fname : string, optional 
            The ``columns_to_keep_ascii_fname`` string is 
            the filename storing ascii data that 
            determines the ``columns_to_keep`` variable. 
            So ``columns_to_keep_ascii_fname`` is just a convenient way to 
            determine ``columns_to_keep`` that also helps 
            keep a permanent record of the choices you made to process your catalog. 

            The number of rows of data in the ``columns_to_keep_ascii_fname`` file 
            determines the number of columns of halo catalog ASCII data that will be kept. 
            Each row in the file should have 3 columns, 
            one column for each of the three 
            tuple elements in ``columns_to_keep``. 
            The file may begin with any number of header lines beginning with '#'. 
            These will be ignored by Halotools but can be used 
            to provide notes for your own bookkeeping. 

            See halotools/data/RockstarHlistReader_input_example.dat 
            for an explicit example. 

        row_cuts : list, optional 
            List of tuples used to define which rows of the ASCII data will be kept.
            Default behavior is to make no cuts. 

            The row-cut is determined from a list of tuples as follows. 
            Each element of the ``row_cuts`` list is a three-element tuple. 
            The first tuple element must be a string that will 
            be interpreted as the halo property  
            upon which your cut is made. 
            The second and third tuple elements will be interpreted 
            as lower and upper bounds on this halo property, respectively. 

            For example, if you only want to keep halos 
            with :math:`M_{\\rm peak} > 1e10`, 
            then you would set row_cuts = [('halo_mpeak', 1e10, float("inf"))]. 

            Every entry appearing in ``row_cuts`` must also 
            appear in ``columns_to_keep``: for purposes of good bookeeping, 
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

        self.fname = self._get_fname(input_fname)

        self.header_char = self._get_header_char(header_char)

        self._determine_compression_safe_file_opener()

        self.num_cols_total = self.infer_number_of_columns()

        self.columns_to_keep = self._determine_columns_to_keep(**kwargs)
        self._interpret_column_indices_to_keep()
        self._interpret_input_dt()

        input_row_cuts = self._interpret_input_row_cuts(**kwargs)
        self.row_cuts = self._get_row_cuts(input_row_cuts)


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
        finally:
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



