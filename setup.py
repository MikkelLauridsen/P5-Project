from setuptools import setup
from Cython.Build import cythonize

setup(
    name='P5-Project',
    version='',
    # packages=[''],
    url='',
    license='',
    author='',
    author_email='',
    description='',
    ext_modules=cythonize("features.pyx"),
    install_requires=['Cython']
)
