from setuptools import setup, find_packages

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
    scripts=[
        "bin/CAL_7_PERFORM_CAL.py",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "deap",
        "lisflood-model"
    ],
    tests_require=[
        "pytest",
        'lisflood-utilities'
    ],
)
