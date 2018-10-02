# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("lidar/core/ctools",
        sources=["lidar/core/ctools.pyx"],
        # libraries=["m"]  # Unix-like specific
        )
]

setup(
    name='lidar',
    version='0.0.1',
    author='Julian Hernandez Velasquez',
    author_email='jhernandezv@unal.edu.co',
    packages=['lidar'],
    # package_data={'lidar':['Nivel.py','SqlDb.py','static.py','information.py']},
    url='https://github.com/julhernandezv/Lidar.git',
    license='LICENSE.txt',
    description="Class for manipulating SIATA's Scanning Lidar",
    long_description=open('README.md').read(),
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()],
)


# setup(
#     ext_modules=cythonize("tools.pyx"),
# )
