from setuptools import setup
import os

setup(scripts=[
        *[os.path.join('bin', f) for f in os.listdir('bin') if os.path.isfile(os.path.join('bin', f))],
        os.path.join('integration', 'benchmark.py'),
        os.path.join('integration', 'test_synthetic.py')
    ])