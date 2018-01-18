"""Setup script for deeone."""

import re

from setuptools import find_packages
from setuptools import setup

project_name = 'mini_mnist'

setup(
    name=project_name,
    version='0.1',
    include_package_data=True,
    packages=find_packages(),
    description='Mini MNIST example',
)
