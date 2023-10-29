# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:45:12 2022

@author: I.Azuma
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include # to use cimport

ext = Extension("_pam", sources=["_pam.pyx", "gamma.c"], include_dirs=['.', get_include()])
setup(name="_pam", ext_modules=cythonize([ext]))
