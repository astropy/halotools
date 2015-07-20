# -*- coding: utf-8 -*-
"""
Methods and classes to read ASCII files storing simulation data. 

"""

__all__ = ['BehrooziASCIIReader']

from . import catalog_manager, supported_sims, sim_defaults

from ..halotools_exceptions import UnsupportedSimError, CatalogTypeError, HalotoolsCacheError, HalotoolsIOError

class BehrooziASCIIReader(object):
    """ Class containing methods used to read raw ASCII data generated with Rockstar
    and made publicly available by Peter Behroozi. 

    Each new raw halo catalog must be processed with its own instance of this class. 
    """

    def __init__(self, input_fname, simname, halo_finder, redshift, 
        recompress = True, **kwargs):
        """
        Parameters 
        -----------
        input_fname : string 
            Name of the file (including absolute path) to be processed. 

        simname : string 
            Nickname of the simulation, e.g. `bolshoi`. 

        halo_finder : string 
            Nickname of the halo-finder, e.g. `rockstar`. 

        redshift : float 
            Value of the redshift of the snapshot

        cuts_funcobj : function object, optional
            Function used to apply cuts to the rows of the ASCII data. 
            `cuts_funcobj` should accept a structured array as input, 
            and return a boolean array of the same length. 
            If None, default cut is set by `default_halocat_cut`. 
            If set to the string ``nocut``, all rows will be kept. 
            The `cuts_funcobj` must be a callable function defined 
            within the namespace of the `RockstarReader` instance, and 
            it must be a stand-alone function, not a bound method of 
            some other class.  

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

        self.halocat = supported_sims.HaloCatalog(simname, halo_finder, redshift)

        self._process_cuts_funcobj(**kwargs)

    def _process_cuts_funcobj(self, **kwargs):
        """
        """
        if 'cuts_funcobj' in kwargs.keys():
            if kwargs['cuts_funcobj'] == 'nocut':
                g = lambda x : np.ones(len(x), dtype=bool)
                self.cuts_funcobj = g
                self._cuts_description = 'nocut'
            else:
                if callable(kwargs['cuts_funcobj']):
                    self.cuts_funcobj = kwargs['cuts_funcobj']
                    self._cuts_description = 'User-supplied cuts_funcobj'
                else:
                    raise TypeError("The input cuts_funcobj must be a callable function")
                    
        else:
            self.cuts_funcobj = self.default_halocat_cut
            self._cuts_description = 'Default cut set by default_halocat_cut'


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

        return x['mpeak'] > self.halocat.particle_mass*sim_defaults.Num_ptcl_requirement

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
            pass

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
        cuts_funcobj : function object, optional keyword argument
            Function used to determine whether a row of the raw 
            halo catalog is included in the reduced binary. 
            Input of the `cuts_funcobj1 must be a structured array 
            with some subset of the field names of the 
            halo catalog. Output of the `cuts_funcobj` must 
            be a boolean array of length equal to the length of the 
            input structured array. 
            Default is set by the `default_halocat_cut` method. 

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
                        print ncols
                    print chunk[-1]
                    raise HalotoolsIOError("Number of columns does not match length of dtype")

                container.append(a[self.cuts_funcobj(a)])
                chunk = []

        a = np.array(chunk, dtype = dt)
        container.append(a[self.cuts_funcobj(a)])

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

        #self._compress_ascii()

        return output
