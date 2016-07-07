""" Module contains functions used to guarantee that only American spellings
are used throughout the package.
"""

import os
import fnmatch

from .test_amurrica import source_code_string_generator
from ..custom_exceptions import HalotoolsError

__all__ = ('test_halotools_usage_of_np_random_seed', )

msg = ("The test suite detected a call to np.random.seed in the following filename:\n"
    "{0} \nCalls to np.random.seed choose the random number seed \n"
    "by setting a global environment variable, \n"
    "which is not good coding practice for Halotools source code.\n"
    "Any time you use a np.random function, you should instead call that function \n"
    "within the namespace of the astropy.utils.misc.NumpyRNGContext context manager.\n"
    "The NumpyRNGContext context manager sets the number seed to that of its input variable, \n"
    "calls the requested np.random function, and then returns the value of the global seed \n"
    "to whatever value it had prior to the function call.\n"
    "See the Astropy documentation for further information.\n"
    "Example usages of NumpyRNGContext appear throughout Halotools, and also below:\n\n"
    ">>> bad_random_x_position = np.random.uniform(0, 250, 100) # BAD \n\n"
    ">>> from astropy.utils.misc import NumpyRNGContext \n"
    ">>> fixed_seed = 43 \n"
    ">>> good_random_x_position = with NumpyRNGContext(fixed_seed): np.random.uniform(0, 250, 100) # GOOD \n\n")


def filtered_filename_generator(filepat, top):
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            if 'test_seed' not in name:
                yield os.path.join(path, name)


def test_halotools_usage_of_np_random_seed():
    dirname_current_module = os.path.dirname(os.path.realpath(__file__))
    base_dirname = os.path.dirname(dirname_current_module)
    source_code_file_generator = filtered_filename_generator('*.py', base_dirname)

    for fname in source_code_file_generator:
        for i, line in source_code_string_generator(fname):
            line = line.lower()
            if "np.random.seed" in line:
                raise HalotoolsError(msg.format(fname))
