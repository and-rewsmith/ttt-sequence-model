from setuptools import find_packages, setup  # type: ignore

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ttt',
    version='0.1.0',
    packages=find_packages(),
    install_requires=required,
)
