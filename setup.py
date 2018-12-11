# -*- coding: utf-8 -*-
from Cython.Build import cythonize
from setuptools import setup, Extension,find_packages
import numpy as np

#requirements = read requirements.txt

ext_module = [  Extension(name="cytools",
        sources=["eda/_libs/cytools.pyx"],
        include_dirs=[np.get_include()],
        # libraries=["m"]  # Unix-like specific
        ),
]
# ext_module.append(
#     Extension(name="eda.lidar",
#         sources=["eda/rs/lidar.py"],
#         # include_dirs=[np.get_include()],
#     )
# )


setup(
    name='EDA',
    version='0.0.1',
    author='Julian Hernandez Velasquez',
    author_email='jhernandezv@unal.edu.co',
    packages=find_packages(include=['eda','eda.*']),
    # package_dir={'lidar':'eda.rs.lidar','plotbook':'eda.core.plotbook'},
    package_data={'eda':['staticfiles/*']},
    url='https://github.com/julhernandezv/Lidar.git',
    license='LICENSE.txt',
    description="Package for manipulating SIATA's Scanning Lidar",
    long_description=open('README.md').read(),
    ext_modules=cythonize(ext_module),
    #install_requires=requirements,
)


# setup(
#     ext_modules=cythonize("tools.pyx"),
# )
zip_safe=True
