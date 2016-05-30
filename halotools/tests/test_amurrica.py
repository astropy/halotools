""" Module contains functions used to guarantee that only American spellings
are used throughout the package.
"""

import os
import fnmatch

from ..custom_exceptions import AmurricaError
from . import usa


def source_code_string_generator(fname):
    """ Yield each line of source code.
    Each line will be checked for all misspellings.
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            yield i, l


def filtered_filename_generator(filepat, top):
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            if 'usa.py' not in name:
                yield os.path.join(path, name)


def test_usa():

    dirname_current_module = os.path.dirname(os.path.realpath(__file__))
    base_dirname = os.path.dirname(dirname_current_module)
    source_code_file_generator = filtered_filename_generator('*.py', base_dirname)

    offensive_spellings = usa.misspellings

    for fname in source_code_file_generator:
        for i, line in source_code_string_generator(fname):
            line = line.lower()
            for t in offensive_spellings:
                if t[1] in line:
                    basename = os.path.basename(fname)
                    raise AmurricaError(basename, i, t[0], t[1])
