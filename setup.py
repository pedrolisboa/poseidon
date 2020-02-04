from pip.req import parse_requirements
from setuptools import setup, find_packages

setup(
    name="poseidon",
    version="0.1",
    packages=find_packages(),
    description="Toolkit for sonar signal processing",
    install_reqs = parse_requirements('requirements.txt')
)