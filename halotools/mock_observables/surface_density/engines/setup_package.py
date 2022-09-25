from distutils.extension import Extension
import os
import numpy as np

PATH_TO_PKG = os.path.relpath(os.path.dirname(__file__))
SOURCES = (
    "weighted_npairs_xy_engine.pyx",
    "weighted_npairs_per_object_xy_engine.pyx",
    "mean_delta_sigma_engine.pyx",
)

THIS_PKG_NAME = ".".join(__name__.split(".")[:-1])


def get_extensions():

    names = [THIS_PKG_NAME + "." + src.replace(".pyx", "") for src in SOURCES]
    sources = [os.path.join(PATH_TO_PKG, srcfn) for srcfn in SOURCES]
    include_dirs = [np.get_include()]
    libraries = []
    language = "c++"
    extra_compile_args = ["-Ofast"]

    extensions = []
    for name, source in zip(names, sources):
        extensions.append(
            Extension(
                name=name,
                sources=[source],
                include_dirs=include_dirs,
                libraries=libraries,
                language=language,
                extra_compile_args=extra_compile_args,
            )
        )

    return extensions
