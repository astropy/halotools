

from distutils.extension import Extension
import os
import sys

PATH_TO_PKG = os.path.relpath(os.path.dirname(__file__))
SOURCES = ["cpairs.pyx", "distances.pyx", "pairwise_distances.pyx"]
THIS_PKG_NAME = '.'.join(__name__.split('.')[:-1])

def get_extensions():

    names = [THIS_PKG_NAME + "." + src.replace('.pyx', '') for src in SOURCES]
    sources = [os.path.join(PATH_TO_PKG, srcfn) for srcfn in SOURCES]
    include_dirs = ['numpy']
    libraries = []
    language ='c++'
    extra_compile_args = []

    extensions = []
    for name, source in zip(names, sources):
        extensions.append(Extension(name=name,
                          sources=[source],
                          include_dirs=include_dirs,
                          libraries=libraries,
                          language = language,
                          extra_compile_args=extra_compile_args))

    return extensions
