from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [
              Extension("sinha_pair_counter_wrapper", ["sinha_pair_counter_wrapper.pyx"],
              include_dirs =["../../../../cextern/sinha_pair_counter/"],
              library_dirs =["../../../../cextern/sinha_pair_counter/"],
              library=["contpairs.h"])
]

extra_compile_args = ["-stdlib=libc++"]

setup(
    name = "sinha_pair_counter_wrapper app",
    ext_modules = cythonize(extensions),
)

#to compile code type:
#    CC=clang python setup.py build_ext --inplace