# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='sumap',
    version='0.0.0.dev8',
    description='SUMAP: Supervised UMAP',
    long_description=readme,
    # to render md at PyPi
    long_description_content_type='text/markdown',
    author='Tianlin He',
    author_email='tinaho_ok@hotmail.com',
    url='https://github.com/tianlinhe/sumap',
    license=license,
    # 1) when ./ contains single .py
    # py_modules=['sumap'],
    # 2) When ./ contains directories
    # packages=find_packages(exclude=('tests',
    #                                 'docs',
    #                                 'examples',
    #                                 'figures')),
    # Or simply
    packages=["sumap"],
    install_requires=[
        "numpy >= 1.17",
        "scikit-learn >= 0.22",
        "pynndescent >= 0.5",
        "scipy >= 1.3.1",
        "numba >= 0.51.2",
        "pynndescent >= 0.5",
        "tbb >= 2019.0",
        "umap-learn >= 0.5.1"
        ],
    )
