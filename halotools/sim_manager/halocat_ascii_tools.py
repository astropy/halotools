import numpy as np
import os

import sim_specs 


class RockstarReader(object):

    def __init__(self, input_fname, catobj):

        if not os.path.isfile(input_fname):
            raise IOError("Input filename %s is not a file" % input_fname)
        self.fname = self._unzip_ascii(input_fname)

        if not isistance(catobj, sim_specs.HaloCatSpecs):
            raise IOError("Input catalog object %s "
                "must be a subclass of HaloCatSpecs" % catobj.__name__)
        self.catobj = catobj

        self.halocat_reader = self.get_raw_halocat_reader()

    def get_raw_halocat_reader(self):
        """ Find and return the class instance that will be used to 
        convert raw ASCII halo catalog data into a reduced binary.

        Returns 
        -------
        halocat_reader : object 
            Class instance of `~halotools.sim_manager.sim_specs.HaloCatSpecs`. 
            Used to read ascii data in the specific format of the 
            `simname` simulation and `halo_finder` halos. 
        """
        simname = self.catobj.simname
        halo_finder = self.catobj.halo_finder

        class_list = sim_specs.__all__
        parent_class = sim_specs.HaloCatSpecs

        supported_halocat_classes = []
        for clname in class_list:
            clobj = getattr(sim_specs, clname)
            if (issubclass(clobj, parent_class)) & (clobj.__name__ != parent_class.__name__):
                supported_halocat_classes.append(clobj())

        halocat_reader = None
        for reader in supported_halocat_classes:
            if (reader.simname == simname) & (reader.halo_finder == halo_finder):
                halocat_reader = reader
        if halocat_reader==None:
            print("No reader class found for %s simulation and %s halo-finder\n"
                "If you want to use Halotools to convert a raw halo catalog into a binary, \n"
                "you must either use an existing reader class or write your own\n" 
                % simname, halo_finder)
            return None
        else:
            return halocat_reader


    def file_len(self):
        """ Compute the number of all rows in fname

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
        """ Compute the number of header rows in fname. 

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

    def get_header(self, Nrows_header_total=None):
        """ Return the header as a list of strings, 
        one entry per header row. 

        Parameters 
        ----------
        fname : string 

        Nrows_header_total :  int, optional
            If the total number of header rows is not known in advance, 
            method will call `header_len` to determine Nrows_header_total. 

        Notes 
        -----
        Empty lines will be included in the returned header. 

        """

        if Nrows_header_total==None:
            Nrows_header_total = self.header_len(self.fname)

        print("Reading the first %i lines of the ascii file" % Nrows_header_total)

        output = []
        with open(self.fname) as f:
            for i in range(Nrows_header_total):
                line = f.readline().strip()
                output.append(line)

        return output

    def _unzip_ascii(self, fname):
        """ If the input fname has file extension `.gz`, 
        then the method uses `gunzip` to decompress it, 
        and returns the input fname truncated to exclude 
        the `.gz` extension. If the input fname does not 
        end in `.gz`, method does nothing besides return 
        the input fname. 
        """
        if fname[-3:]=='.gz':
            os.system("gunzip "+fname)
            return fname[:-3]
        else:
            return fname

    def _rezip_ascii(self, fname):
        """ Recompresses the halo catalog ascii data, 
        and returns the input filename appended with `.gz`.  
        """
        os.system("gzip "+fname)
        return fname+'.gz'


    def reader(self, dt, cut, Nchunks = 1000):
        """ Reads fname in chunks and returns a structured array
        after applying cuts.
        """

        file_length = self.file_len(self.fname)
        chunksize = file_length / Nchunks
        chunk = []
        container = []
        for linenum, line in enumerate(open(self.fname)):
            if line[0] == '#':
                pass
            else:
                parsed_line = line.strip().split()
                chunk.append(tuple(parsed_line))  
        
            if (linenum % chunksize == 0) & (linenum > 0):        
                a = np.array(chunk, dtype = dt)
                container.append(a[cut(a)])
                chunk = []
    #Now for the final chunk missed by the above syntax
        a = np.array(chunk, dtype = dt)
        container.append(a[cut(a)])

    # Bundle up all array chunks into a single array
        for chunk in container:
            try:
                output = np.append(output, chunk)
            except NameError:
                output = np.array(chunk) 
                
        return output



