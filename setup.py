from setuptools import setup, find_packages
import os

exec(open('liscal/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="liscal",
    version=__version__,
    author='JRC-ECMWF',
    description="LISFLOOD calibration tool",
    long_description=long_description,    
    packages=find_packages(exclude=["test_*", "*.tests", "*.tests.*", "tests.*", "tests"]),
    scripts=[*[os.path.join('bin', i) for i in os.listdir('bin')],
        os.path.join('integration', 'benchmark.py'),
        os.path.join('integration', 'test_synthetic.py')],
    install_requires=[
        "numpy",
        "pandas",
        "dask",
        "distributed",
        "xarray",
        "deap",
        "mpi4py",
        #"pcraster",
        # "lisflood-model"
    ],
    tests_require=[
        "pytest",
        "gzip",
    ],
)
