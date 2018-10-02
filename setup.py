# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# ext_modules = [
#     Extension("tools",
#         sources=["tools.pyx"],
#         # libraries=["m"]  # Unix-like specific
#         )
# ]

setup(
ext_modules=cythonize("lidar/core/ctools.pyx"),
    include_dirs=[np.get_include()],
)


# setup(
#     ext_modules=cythonize("tools.pyx"),
# )
