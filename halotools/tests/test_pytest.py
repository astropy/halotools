"""
"""
import os
import fnmatch

from .test_amurrica import source_code_string_generator
from ..custom_exceptions import HalotoolsError

__all__ = ('test_halotools_pytest_imports', )

msg = ("\nHalotools has the following development requirement for all modules in the repo:\n"
    "It is not permissible to directly ``import pytest``, \n"
    "as is currently done in the following file:\n\n{0}\n\n"
    "Instead, use ``from astropy.tests.helper import pytest`` \n")


def filtered_filename_generator(filepat, top):
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            if 'test_pytest' not in name:
                yield os.path.join(path, name)


def test_halotools_pytest_imports():
    dirname_current_module = os.path.dirname(os.path.realpath(__file__))
    base_dirname = os.path.dirname(dirname_current_module)
    source_code_file_generator = filtered_filename_generator('*.py', base_dirname)

    for fname in source_code_file_generator:
        for i, line in source_code_string_generator(fname):
            line = line.lower()
            if "import pytest" in line:
                try:
                    assert "from astropy.tests.helper" in line
                except AssertionError:
                    raise HalotoolsError(msg.format(fname))
