from distutils.extension import Extension
import os

PATH_TO_WRAPPER = os.path.relpath(os.path.dirname(__file__))
SOURCES = ["sinha_pairs.pyx", "source/countpairs.c", "source/gridlink.c", "source/utils.c"]
THIS_PKG_NAME = '.'.join(__name__.split('.')[:-1])

def get_extensions():
    name = THIS_PKG_NAME + "." + SOURCES[0].replace('.pyx', '')
    sources = [os.path.join(PATH_TO_WRAPPER, srcfn) for srcfn in SOURCES]
    include_dirs = [os.path.join(PATH_TO_WRAPPER, 'include')]
    libraries = []
    extra_compile_args = ["-DPERIODIC", "-DDOUBLE_PREC"]

    extensions = [Extension(name=name,
                  sources=sources,
                  include_dirs=include_dirs,
                  libraries=libraries,
                  extra_compile_args=extra_compile_args)
                 ]

    return extensions
