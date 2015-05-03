import numpy as np
import os

from . sim_specs import HaloCatSpecs


class RockstarReader(object):

    def __init__(self, input_fname, catobj):

        if not os.path.isfile(input_fname):
            raise IOError("Input filename %s is not a file" % input_fname)
        self.fname = self._unzip_ascii(input_fname)

        if not isistance(catobj, HaloCatSpecs):
            raise IOError("Input catalog object %s "
                "must be a subclass of HaloCatSpecs" % catobj.__name__)
        self.catobj = catobj

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



