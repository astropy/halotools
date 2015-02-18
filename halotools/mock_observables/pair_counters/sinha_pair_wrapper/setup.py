from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([Extension("sinha_pair_counter_wrapper", ["sinha_pair_counter_wrapper.pyx"], libraries =["countpairs"], library_dirs=["../../../../cextern/sinha_pair_counter"])])
)