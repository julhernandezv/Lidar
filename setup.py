# -*- coding: utf-8 -*-
from Cython.Build import cythonize
from setuptools import setup, Extension,find_packages
import numpy as np

#requirements = read requirements.txt

ext_module = [  Extension(name="cytools",
        sources=["lidar/_libs/cytools.pyx"],
        include_dirs=[np.get_include()],
        # libraries=["m"]  # Unix-like specific
        ),
]



setup(
    name='lidar',
    version='0.0.1',
    author='Julian Hernandez Velasquez',
    author_email='jhernandezv@unal.edu.co',
    packages=find_packages(include=['lidar','lidar.*']),
    package_dir={'lidar':'lidar'},
    package_data={'lidar':['staticfiles/*']},
    url='https://github.com/julhernandezv/Lidar.git',
    license='LICENSE.txt',
    description="Class for manipulating SIATA's Scanning Lidar",
    long_description=open('README.md').read(),
    ext_modules=cythonize(ext_module),
    #install_requires=requirements,
)


# setup(
#     ext_modules=cythonize("tools.pyx"),
# )
zip_safe=True
