from setuptools import setup, find_packages

setup(
    name="gobc_web",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pytest",
        "pytest-asyncio",
    ],
) 