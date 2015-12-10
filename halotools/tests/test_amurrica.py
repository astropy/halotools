""" Module contains functions used to guarantee that only American spellings 
are used throughout the package. 
""" 

import os, fnmatch

from ..custom_exceptions import AmurricaError, HalotoolsError

def offensive_spellings_generator():
    """ Look in usa.dat for two-element tuples containing the misspellings. 
    """
    dirname_current_module = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dirname_current_module, 'usa.py')

    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            pair = line.strip().split()
            yield pair

def source_code_string_generator(fname):
    """ Yield each line of source code. 
    Each line will be checked for all misspellings. 
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            yield i, l

def gen_find(filepat, top):
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist,filepat):
            if 'usa.py' not in name:
                yield os.path.join(path,name)

def test_usa():

    dirname_current_module = os.path.dirname(os.path.realpath(__file__))
    base_dirname = os.path.dirname(dirname_current_module)
    source_code_file_generator = gen_find('*.py', base_dirname)

    offensive_spelling_list = list(offensive_spellings_generator())

    for fname in source_code_file_generator:
      for i, line in source_code_string_generator(fname):
          for t in offensive_spelling_list:
              if t[1] in line:
                basename = os.path.basename(fname)
                raise AmurricaError(basename, i, t[0], t[1])






