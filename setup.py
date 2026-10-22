# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='Atmosphere analysis project',
    version='0.1.0',
    description='Atmosphere analysis project',
    long_description=readme,
    author='Ruiyao Chen',
    author_email='Ruiyao.Chen@vanderbilt.edu',
    url='https://github.com/ruiyao-chen/precip_classify',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

