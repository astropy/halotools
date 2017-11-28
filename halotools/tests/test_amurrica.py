""" Module contains functions used to guarantee that only
correct American spellings are used throughout the Halotools source code.
"""

import os
import fnmatch

from ..custom_exceptions import AmurricaError, SpellingError
from . import usa, ddc


def test_amurrica_instantiation():
    try:
        basename, linenum, correct_spelling, offending_spelling = 'modulename', 20, 'president', 'queen'
        raise AmurricaError(basename, linenum, correct_spelling, offending_spelling)
    except AmurricaError:
        pass


def test_spelling_instantiation():
    try:
        basename, linenum, correct_spelling, offending_spelling = 'modulename', 20, 'Arnold', 'Sly'
        raise SpellingError(basename, linenum, correct_spelling, offending_spelling)
    except SpellingError:
        pass


def source_code_string_generator(fname):
    """ Yield each line of source code.
    Each line will be checked for all misspellings.
    """
    with open(fname) as f:
        for i, l in enumerate(f):
            yield i, l


def filtered_filename_generator(filepat, top, fname_to_skip):
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            if fname_to_skip not in name:
                yield os.path.join(path, name)


def test_usa():

    dirname_current_module = os.path.dirname(os.path.realpath(__file__))
    base_dirname = os.path.dirname(dirname_current_module)
    source_code_file_generator = filtered_filename_generator('*.py', base_dirname, 'usa.py')

    offensive_spellings = usa.misspellings

    for fname in source_code_file_generator:
        for i, line in source_code_string_generator(fname):
            line = line.lower()
            for t in offensive_spellings:
                if t[1] in line:
                    basename = os.path.basename(fname)
                    raise AmurricaError(basename, i, t[0], t[1])


def test_common_contributor_misspellings():

    dirname_current_module = os.path.dirname(os.path.realpath(__file__))
    base_dirname = os.path.dirname(dirname_current_module)
    source_code_file_generator = filtered_filename_generator('*.py', base_dirname, 'ddc.py')

    for fname in source_code_file_generator:
        for i, line in source_code_string_generator(fname):
            line = line.lower()
            for t in ddc.misspellings:
                if t[1] in line:
                    basename = os.path.basename(fname)
                    raise SpellingError(basename, i, t[0], t[1])
