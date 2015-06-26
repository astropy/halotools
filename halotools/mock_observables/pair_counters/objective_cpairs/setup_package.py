from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import os


dir = os.path.dirname(__file__)
if dir=='': dir='./'

PATH_TO_WRAPPER = os.path.relpath(dir)
SOURCES = ["objective_cpairs.pyx","objective_weights.pyx"]
THIS_PKG_NAME = '.'.join(__name__.split('.')[:-1])

def main():
    """
    This main function can be used to build this module in place
    run command: python setup_package.py build_ext --inplace
    """
    setup(ext_modules = cythonize(SOURCES),include_dirs = [np.get_include()])


def get_extensions():
     
    name = THIS_PKG_NAME + "." + SOURCES[0].replace('.pyx', '')
    sources = [os.path.join(PATH_TO_WRAPPER, srcfn) for srcfn in SOURCES]
    include_dirs = [np.get_include(),PATH_TO_WRAPPER]
    libraries = []
    extra_compile_args = []
    print(name,sources)

    extensions = [Extension(name=name,
                  sources=sources,
                  include_dirs=include_dirs,
                  libraries=libraries,
                  extra_compile_args=extra_compile_args)
                 ]
    
    name = THIS_PKG_NAME + "." + SOURCES[1].replace('.pyx', '')
    sources = [os.path.join(PATH_TO_WRAPPER, srcfn) for srcfn in SOURCES]
    include_dirs = [np.get_include()]
    libraries = []
    extra_compile_args = []
    print(name,sources)

    extensions.append(Extension(name=name,
                  sources=sources,
                  include_dirs=include_dirs,
                  libraries=libraries,
                  extra_compile_args=extra_compile_args)
                 )
                 
    return extensions


if __name__ == '__main__':
    main()