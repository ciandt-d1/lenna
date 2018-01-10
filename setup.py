"""Setup script for deeone."""

import re

from setuptools import find_packages
from setuptools import setup

project_name = 'tf_image_classification'

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  open('{}/__init__.py'.format(project_name)).read()
).group(1)

setup(
    name=project_name,
    version=__version__,
    include_package_data=True,
    packages=find_packages(),
    description='tensorflow image classification framework',
    install_requires=[
        "google-cloud-storage==1.6.0",
    ]
)
