# -*- coding: utf-8 -*-
"""
Methods and classes to read ASCII files storing simulation data. 

"""

__all__ = ['BehrooziASCIIReader']

import os
from time import time
import numpy as np
from difflib import get_close_matches

from astropy.table import Table

from . import catalog_manager, supported_sims, sim_defaults, cache_config

from ..custom_exceptions import UnsupportedSimError, CatalogTypeError, HalotoolsCacheError, HalotoolsIOError

class BehrooziASCIIReader(object):
    """ Class containing methods used to read raw ASCII data generated with Rockstar 
    and made publicly available by Peter Behroozi. 
    """
    def __init__(self, input_fname, recompress = True, **kwargs):
        """
        Parameters 
        -----------
        input_fname : string 
            Name of the file (including absolute path) to be processed. 

        simname : string, optional 
            Nickname of the simulation, e.g. ``bolshoi``. If None is passed, 
            Halotools will attempt to automatically infer this from ``input_fname``. 

        halo_finder : string, optional 
            Nickname of the halo-finder, e.g. ``rockstar``. If None is passed, 
            Halotools will attempt to automatically infer this from ``input_fname``. 

        cuts_funcobj : function object, optional
            Any function used to apply row-wise cuts when reading ASCII data. 
            ``cuts_funcobj`` should accept a structured array or Astropy Table as input, 
            and return a boolean array of the same length. 
            If None, default cut is set by ``default_halocat_cut``. 
            If set to the string ``nocut``, all rows will be kept. 
            The ``cuts_funcobj`` must be a callable function defined 
            within the namespace of the `RockstarReader` instance, and 
            it must be a stand-alone function, not a bound method of 
            some other class.  
            If passing ``cuts_funcobj`` keyword argument, 
            you may not pass a ``column_bounds`` keyword argument. 

        column_bounds : list, optional 
            List of tuples used to apply row-wise cuts when reading ASCII data. 
            Each list element is a three-element tuple. The first tuple element 
            must be a column key of the halo catalog. The second and third 
            tuple elements will be interpreted as lower and upper bounds. 
            If column_bounds is an N-element list, only ASCII rows passing *all* 
            cuts will be kept in the resulting catalog. 
            If passing ``column_bounds`` keyword argument, 
            you may not pass a ``cuts_funcobj`` keyword argument. 

        recompress : bool, optional 
            If ``input_fname`` is a compressed file, `BehrooziASCIIReader` 
            will automatically uncompress it before reading. If recompress 
            is True, the file will be recompressed after reading; if False, 
            it will remain uncompressed. Default is True. 
        """

        # Check whether input_fname exists. 
        # If not, raise an exception. If so, bind to self. 
        if not os.path.isfile(input_fname):
            # Check to see whether the uncompressed version is in cache
            if not os.path.isfile(input_fname[:-3]):
                msg = "Input filename %s is not a file" 
                raise HalotoolsCacheError(msg % input_fname)
            else:
                msg = ("Input filename ``%s`` is not a file. \n"
                    "However, ``%s`` is, so change your input_fname accordingly.")
                raise HalotoolsCacheError(msg % (input_fname, input_fname[:-3]))
        else:
            self.fname = input_fname
                
        self._recompress = recompress
        self._uncompress_ascii()

        self.catman = catalog_manager.CatalogManager()

        simname, halo_finder, redshift = self._infer_snapshot(self.fname, **kwargs)
        self.halocat = supported_sims.HaloCatalog(
            simname=simname, halo_finder=halo_finder, desired_redshift=redshift)

        self._process_cuts_funcobj(**kwargs)

    def _infer_snapshot(self, fname, **kwargs):
        """
        """

        subdir = os.path.abspath(os.path.join(fname, os.pardir))
        pardir = os.path.abspath(os.path.join(subdir, os.pardir))

        if 'halo_finder' in kwargs.keys():
            halo_finder = kwargs['halo_finder']
        else:
            halo_finder = os.path.basename(subdir)

        if 'simname' in kwargs.keys():
            simname = kwargs['simname']
        else:
            simname = os.path.basename(pardir)

        if simname not in cache_config.supported_sim_list:
            msg = "Halotools tried to infer your simname from the input fname = %s \n"
            "The inferred simname is %s, which is not recognized.\n "
            "Try calling the function again but using ``simname`` as a keyword argument, \n"
            "or otherwise add your simname to the list of supported sims. "
            raise HalotoolsCacheError(msg % (fname, simname))

        supported_halo_finders = cache_config.get_supported_halo_finders(simname)
        if halo_finder not in supported_halo_finders:
            msg = "Halotools tried to infer your halo_finder from the input fname = %s \n"
            "The inferred halo_finder is %s, which is not recognized.\n "
            "Try calling the function again but using ``halo_finder`` as a keyword argument, \n"
            "or otherwise add your halo_finder to the list of supported halo_finders. "
            raise HalotoolsCacheError(msg % (fname, halo_finder))

        scale_factor = float(self.catman._get_scale_factor_substring(os.path.basename(fname)))
        redshift = (1./scale_factor) - 1

        return simname, halo_finder, redshift


    def _process_cuts_funcobj(self, **kwargs):
        """
        """
        if ('cuts_funcobj' in kwargs.keys()) & ('column_bounds' in kwargs.keys()):
            raise KeyError("You may not pass both a ``cuts_funcobj`` and a ``column_bounds`` argument")
        elif 'cuts_funcobj' in kwargs.keys():
            if kwargs['cuts_funcobj'] == 'nocut':
                g = lambda x : np.ones(len(x), dtype=bool)
                self.cuts_funcobj = g
                self._cuts_description = 'nocut'
            else:
                if callable(kwargs['cuts_funcobj']):
                    self.cuts_funcobj = kwargs['cuts_funcobj']
                    self._cuts_description = ('User-supplied cuts_funcobj '
                        'given as cuts_funcobj keyword argument to BehrooziASCIIReader constructor')
                else:
                    raise TypeError("The input cuts_funcobj must be a callable function")
        elif 'column_bounds' in kwargs.keys():
            column_bounds = kwargs['column_bounds']
            for cut in column_bounds:
                if cut[0] not in self.halocat.dtype_ascii.names:
                    msg = ("columns_bound keyword argument included a cut on ``%s``, "
                        "which is not a column of this halo catalog\n")
                    possible_matches = get_close_matches(cut[0], self.halocat.dtype_ascii.names)
                    if possible_matches != []:
                        s = ''
                        for elt in possible_matches: 
                            s = elt + ', ' + s
                        s = s[:-2]

                        additional_msg = "Did you mean any of the following keys?\n" + s
                        msg = msg + additional_msg
                    raise HalotoolsIOError(msg % cut[0])

            def return_cutfunc(column_bounds):
                cutfuncs = []
                for cut in column_bounds:
                    g = lambda x : (x[cut[0]] > cut[1]) & (x[cut[0]] < cut[2])
                    cutfuncs.append(g)
                def composite_cut(t):
                    result = np.ones(len(t), dtype=bool)
                    for cut in cutfuncs:
                        result *= cut(t)
                    return result
                return composite_cut
            self.cuts_funcobj = return_cutfunc(column_bounds)
            self._cuts_description = ('User-supplied cuts_funcobj '
                'given as cuts_funcobj keyword argument to BehrooziASCIIReader constructor')
        else:
            self.cuts_funcobj = self.default_halocat_cut
            self._cuts_description = 'Default cut set by BehrooziASCIIReader.default_halocat_cut method'


    def default_halocat_cut(self, x):
        """ Function used to provide a simple cut on a raw halo catalog, 
        such that only rows with :math:`M_{\\rm peak} > 300m_{\\rm p}` 
        pass the cut. 

        Parameters 
        ----------
        x : array 
            Length-Nhalos structured numpy array, presumed to have a field called `mpeak`. 

        Returns 
        -------
        result : array
            Length-Nhalos boolean array serving as a mask. 
        """

        key = sim_defaults.mass_like_variable_to_apply_cut
        mp = self.halocat.particle_mass
        nptcl = sim_defaults.Num_ptcl_requirement
        mass_cut = nptcl * mp
        return x[key] > mass_cut

    def file_len(self):
        """ Compute the number of all rows in the raw halo catalog. 

        Parameters 
        ----------
        fname : string 

        Returns 
        -------
        Nrows : int
     
        """
        with open(self.fname) as f:
            for i, l in enumerate(f):
                pass
        Nrows = i + 1
        return Nrows

    def header_len(self,header_char='#'):
        """ Compute the number of header rows in the raw halo catalog. 

        Parameters 
        ----------
        fname : string 

        header_char : str, optional
            string to be interpreted as a header line

        Returns 
        -------
        Nheader : int

        Notes 
        -----
        All empty lines that appear in header 
        will be included in the count. 

        """
        Nheader = 0
        with open(self.fname) as f:
            for i, l in enumerate(f):
                if ( (l[0:len(header_char)]==header_char) or (l=="\n") ):
                    Nheader += 1
                else:
                    break

        return Nheader


    def _uncompress_ascii(self):
        """ If the input fname has file extension `.gz`, 
        then the method uses `gunzip` to decompress it, 
        and returns the input fname truncated to exclude 
        the `.gz` extension. If the input fname does not 
        end in `.gz`, method does nothing besides return 
        the input fname. 
        """
        if self.fname[-3:]=='.gz':
            print("...uncompressing ASCII data")
            os.system("gunzip "+self.fname)
            self.fname = self.fname[:-3]
        else:
            # input_fname was already decompressed. 
            # Turn off auto-recompress
            self._recompress = False

    def _compress_ascii(self):
        """ Recompresses the halo catalog ascii data, 
        and returns the input filename appended with `.gz`.  
        """

        recompress = (self._recompress) & (self.fname[-3:]!='.gz')
        if recompress is True:
            print("...re-compressing ASCII data")
            os.system("gzip "+self.fname)
            self.fname = self.fname + '.gz'


    def read_halocat(self, **kwargs):
        """ Reads the raw halo catalog in chunks and returns a structured array
        after applying cuts.

        Parameters 
        ----------
        nchunks : int, optional keyword argument
            `read_halocat` reads and processes ascii 
            in chunks at a time, both to improve performance and 
            so that the entire raw halo catalog need not fit in memory 
            in order to process it. The total number of chunks to use 
            can be specified with the `nchunks` argument. Default is 1000. 

        """
        start = time()

        # First read the first line as a self-consistency check against self.halocat.header_ascii
        with open(self.fname) as f:
            self._header_ascii_from_input_fname = f.readline()

        if 'nchunks' in kwargs.keys():
            Nchunks = kwargs['nchunks']
        else:
            Nchunks = 1000

        dt = self.halocat.dtype_ascii

        file_length = self.file_len()
        header_length = self.header_len()
        chunksize = file_length / Nchunks
        if chunksize == 0:
            chunksize = file_length # data will now never be chunked
            Nchunks = 1

        print("\n...Processing ASCII data of file: \n%s\n " % self.fname)
        print(" Total number of rows in file = %i" % file_length)
        print(" Number of rows in detected header = %i \n" % header_length)
        if Nchunks==1:
            print("Reading catalog in a single chunk of size %i\n" % chunksize)
        else:
            print("...Reading catalog in %i chunks, each with %i rows\n" % (Nchunks, chunksize))

        print("Applying the following row-wise cuts: \n%s\n" % self._cuts_description)

        chunk_counter = 0
        chunk = []
        container = []
        iout = np.round(Nchunks / 10.).astype(int)
        for linenum, line in enumerate(open(self.fname)):

            if line[0] == '#':
                pass
            else:
                parsed_line = line.strip().split()
                chunk.append(tuple(parsed_line))  
        
            if (linenum % chunksize == 0) & (linenum > 0):

                chunk_counter += 1
                if (chunk_counter % iout)==0:
                    print("... working on chunk # %i of %i\n" % (chunk_counter, Nchunks))

                try:
                    a = np.array(chunk, dtype = dt)
                except ValueError:
                    Nfields = len(dt.fields)
                    print("Number of fields in np.dtype = %i" % Nfields)
                    Ncolumns = []
                    for elt in chunk:
                        if len(elt) not in Ncolumns:
                            Ncolumns.append(len(elt))
                    print("Number of columns in chunk = ")
                    for ncols in Ncolumns:
                        print(ncols)
                    print chunk[-1]
                    raise HalotoolsIOError("Number of columns does not match length of dtype")

                container.append(a[self.cuts_funcobj(a)])
                chunk = []

        a = np.array(chunk, dtype = dt)
        container.append(a[self.cuts_funcobj(a)])

        print("Done reading ASCII. Now bundling into a single array")
    # Bundle up all array chunks into a single array
        for chunk in container:
            try:
                output = np.append(output, chunk)
            except NameError:
                output = np.array(chunk) 
                

        end = time()
        runtime = (end-start)
        if runtime > 60:
            runtime = runtime/60.
            msg = "Total runtime to read in ASCII = %.1f minutes\n"
        else:
            msg = "Total runtime to read in ASCII = %.1f seconds\n"
        print(msg % runtime)

        self._compress_ascii()

        return Table(output)
