from setuptools import setup, find_packages

setup(
    name='pointnet2_ops',
    version='3.0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
)