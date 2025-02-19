# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("p_fast.pyx"),
    include_dirs=[np.get_include()]
)

setup(
    ext_modules=cythonize("b_fast.pyx"),
    include_dirs=[np.get_include()]
)