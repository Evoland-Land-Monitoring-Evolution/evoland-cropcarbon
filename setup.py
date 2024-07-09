#!/usr/bin/env python3

from setuptools import setup, find_packages

# Load the version info.
#
# Note that we cannot simply import the module, since dependencies listed
# in setup() will very likely not be installed yet when setup.py run.
#
# See:
#   https://packaging.python.org/guides/single-sourcing-package-version

__version__ = None

with open('src/cropcarbon/_version.py') as fp:
    exec(fp.read())

# Configure setuptools

setup(
    name='cropcarbon',
    version=__version__,
    author='Kasper Bonte',
    author_email='kasper.bonte@vito.be',
    description='Carbon mapping work',
    url='https://git.vito.be/scm/~bontek/cropcarbon.git',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        '': [
            'static/*',
            'templates/*',
            'resources/*'
        ]
    },
    include_package_data=True
)
