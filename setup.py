import pathlib

import pkg_resources
import setuptools


from setuptools import setup, find_packages
import pip

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name="poseidon",
    version="0.1",
    packages=find_packages(),
    description="Toolkit for sonar signal processing",
    install_requires=install_requires,
)
