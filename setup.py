"""Setup script for deeone."""

import re

from setuptools import find_packages
from setuptools import setup

project_name = 'lenna'

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  open('{}/__init__.py'.format(project_name)).read()
).group(1)

req_list=["google-cloud-storage==1.6.0", "matplotlib", "numpy", "pandas", "scipy", "six", "seaborn", "Sphinx", "sphinx-rtd-theme", "tensorflow==1.8.0"]
 

setup(
    name=project_name,
    version=__version__,
    include_package_data=True,
    packages=find_packages(),
    description='Lenna - Tensorflow image classification framework',
    install_requires=req_list,
    author = 'CI&T'
)
