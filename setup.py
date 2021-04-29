from setuptools import find_packages, setup

setup(
    name='datapointloss',
    packages=find_packages(include=['datapointloss']),
    version='1.0',
    description='Library for drawing brier curves and visualizing the loss of every datapoint',
    author='Magnus Paal',
    license='MIT',
    install_requires=['numpy', 'matplotlib', 'scipy'],
)