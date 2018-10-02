# -*- coding: utf-8 -*-
from Cython.Build import cythonize
# from distutils.core import setup
# from distutils.extension import Extension
from setuptools import setup, Extension,find_packages
import numpy as np

ext_modul = [  Extension("lidar._libs.ctools",
        sources=["lidar/_libs/ctools.pyx"],
        include_dirs=[np.get_include()],
        # libraries=["m"]  # Unix-like specific
        )
]
print find_packages(include=['lidar','lidar.*'])
print ext_modul
print cythonize(ext_modul)
setup(
    name='lidar',
    version='0.0.1',
    author='Julian Hernandez Velasquez',
    author_email='jhernandezv@unal.edu.co',
    packages=find_packages(include=['lidar','lidar.*']),
    # package_data={'lidar':['core.plotbook.py','lidar.py']}, #'core/SqlDb.py'
    url='https://github.com/julhernandezv/Lidar.git',
    license='LICENSE.txt',
    description="Class for manipulating SIATA's Scanning Lidar",
    long_description=open('README.md').read(),
    ext_modules=cythonize(ext_modul), #solo una extención
    # include_dirs=[np.get_include()],
)


# setup(
#     ext_modules=cythonize("tools.pyx"),
# )
