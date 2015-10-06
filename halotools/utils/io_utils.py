# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import sys, urllib

__all__ = ['file_len', 'download_file_from_url']

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1    
    

def download_file_from_url(url, fname):
    """ Function to download a file from the web to a specific location, 
    and print a progress bar along the way. 

    Parameters 
    ----------
    url : string 
        web location of desired file, e.g., 
        ``http://www.some.website.com/somefile.txt``. 

    fname : string 
        Location and filename to store the downloaded file, e.g., 
        ``/Users/username/dirname/possibly_new_filename.txt``
    """

    print("\n... Downloading data from the following location: \n%s\n" % url)
    print(" ... Saving the data with the following filename: \n%s\n" % fname) 

    def reporthook(a,b,c): 
        print("% 3.1f%% of %d bytes\r" % (min(100, float(a * b) / c * 100), c),)
        sys.stdout.flush()

    urllib.urlretrieve(url, fname, reporthook)





