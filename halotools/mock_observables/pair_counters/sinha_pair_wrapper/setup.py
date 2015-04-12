from __future__ import absolute_import, division, print_function

from distutils.extension import Extension
import os
from distutils.core import setup
from Cython.Build import cythonize

PATH_TO_WRAPPER = './'
SOURCES = ["sinha_pairs.pyx", "source/countpairs.c", "source/gridlink.c", "source/utils.c"]

def get_extensions():
    
    name = "sinha_pairs"
    sources = [os.path.join(PATH_TO_WRAPPER, srcfn) for srcfn in SOURCES]
    include_dirs = [PATH_TO_WRAPPER+'include/']
    libraries = []
    extra_compile_args = ["-DPERIODIC","-DDOUBLE_PREC"]
    extra_link_args = []

    extensions= [Extension(name=name, 
                 sources=sources, 
                 include_dirs=include_dirs, 
                 libraries=libraries,
                 extra_compile_args=extra_compile_args,
                 extra_link_args=extra_link_args)
                ]
                
    return extensions

extension = get_extensions()

setup(
    ext_modules = cythonize(extension)
)

os.system('cp sinha_pairs.so ../')

# CC=clang python setup.py build_ext --inplace