"""Minimal setup file for tasks project."""

from setuptools import setup, find_packages

setup(
    name='fdtd',
    version='0.1.0',
    license='proprietary',
    description='Electromagnetic field simulation by fdtd',

    author='Yuta Hirabayashi',

    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    install_requires=['numpy', 'numba']
)