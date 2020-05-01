try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

from setuptools import setup, find_packages
import pip

setup(
    name="poseidon",
    version="0.1",
    packages=find_packages(),
    description="Toolkit for sonar signal processing",
    install_reqs = parse_requirements('requirements.txt')
)
