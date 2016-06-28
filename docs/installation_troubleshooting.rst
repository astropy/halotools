:orphan:

.. _installation_troubleshooting:

****************************************************
Installation Troubleshooting
****************************************************

This page of the documentation collects together solutions
to known installation problems. If you have trouble installing
Halotools and the issue is not resolved here, please contact
the development team and raise an Issue on
`The GitHub Issues page <https://github.com/astropy/halotools/issues>`_.

1. Older versions of the gcc compiler may not support all the optimization
flags that are used to compile the Cython code. These flags are chosen according
to the ``extra_compile_args`` list of strings that appears in the handful of
``setup_package.py`` files throughout the code. If this list contains a flag
not used by your version of gcc, you can simply delete each
appearance of the offending flag, and then install the code by
building the modified source code.
See `GitHub Issue 561 <https://github.com/astropy/halotools/issues/561>`_ for further details.

2. If you are a Mac user and you are not using the version of gcc that ships
with OS X, you may need to temporarily switch compilers to clang.
As described in `GitHub Issue 447 <https://github.com/astropy/halotools/issues/447>`_,
try ``export CC=clang; pip install halotools`` to resolve this issue.

