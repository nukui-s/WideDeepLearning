from setuptools import setup, find_packages

setup(
    name="widedeep",
    version='0.0.0',
    author='nukui',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
)

