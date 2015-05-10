from distutils.extension import Extension
import numpy
import os

PATH_TO_WRAPPER = os.path.relpath(os.path.dirname(__file__))
SOURCES = ["ckdtree.pyx"]
THIS_PKG_NAME = '.'.join(__name__.split('.')[:-1])

def get_extensions():
    name = THIS_PKG_NAME + "." + SOURCES[0].replace('.pyx', '')
    sources = [os.path.join(PATH_TO_WRAPPER, srcfn) for srcfn in SOURCES]
    include_dirs = [numpy.get_include()]
    libraries = []
    extra_compile_args = ["-O3"]

    extensions = [Extension(name=name,
                  sources=sources,
                  include_dirs=include_dirs,
                  libraries=libraries,
                  extra_compile_args=extra_compile_args)
                 ]

    return extensions