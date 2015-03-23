from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize




extensions = [
    Extension("sinha_pair_counter_wrapper", 
    	["sinha_pair_counter_wrapper.pyx"], 
    	libraries =["countpairs"],
    	library_dirs=["/Users/kt/halotools/cextern/sinha_pair_counter"],
    )
]
setup(
    name = "sinha_pair_setup",
    ext_modules = cythonize(extensions),
)
