"""
Copyright 2019-2026 European Union

Licensed under the EUPL, Version 1.2 or as soon they will be approved by the
European Commission subsequent versions of the EUPL (the "Licence");

You may not use this work except in compliance with the Licence.
You may obtain a copy of the Licence at:

https://joinup.ec.europa.eu/sites/default/files/inline-files/EUPL%20v1_2%20EN(1).txt

---------------------------------------------------------------------------------------------------------------------------------------
Versioning
----------
The package version is managed by setuptools_scm and derived from git tags.
- On branches/dev: version is auto-generated (e.g. 2.0.1.dev3+gabc1234).
- For releases: the publish/testpypi commands accept a --version argument
  to create the git tag and build the package with that exact version.

To test pip installation:
    python setup.py testpypi --version 2.0.0rc1
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple liscal==2.0.0rc1

    Note: testpypi creates a temporary local tag for the build, then deletes it.
    No tag is pushed to the remote.

To publish on PyPI:
    python setup.py publish --version 2.0.0

This will:
    1. git tag v{version}          (creates the tag so setuptools_scm picks it up)
    2. python -m build             (builds sdist + wheel with that version)
    3. twine upload dist/*         (uploads to PyPI)
    4. git push --tags             (pushes the tag to the remote)

Install:
    pip install liscal
"""

import os
import sys
from shutil import rmtree

from setuptools import setup, Command

current_dir = os.path.dirname(os.path.abspath(__file__))


def _check_publish_dependencies():
    """Check that 'build' and 'twine' are installed before publishing."""
    missing = []
    if os.system('{} -m build --version > /dev/null 2>&1'.format(sys.executable)) != 0:
        missing.append('build')
    if os.system('{} -m twine --version > /dev/null 2>&1'.format(sys.executable)) != 0:
        missing.append('twine')
    if missing:
        raise SystemExit(
            'Error: missing required packages: {}.\n'
            'Install them with: pip install {}'.format(
                ', '.join(missing), ' '.join(missing)
            )
        )


def _run_or_fail(cmd, error_msg):
    """Run a shell command and abort if it fails."""
    if os.system(cmd) != 0:
        raise SystemExit('Error: {}'.format(error_msg))


class UploadCommand(Command):
    """Build, tag, and publish liscal package to PyPI."""

    description = 'Build and publish liscal package to PyPI.'
    user_options = [
        ('version=', 'V', 'Version to publish (e.g. 2.0.0). Required.'),
    ]

    @staticmethod
    def print_console(s):
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        self.version = None

    def finalize_options(self):
        if self.version is None:
            raise SystemExit(
                'Error: --version is required.\n'
                'Usage: python setup.py publish --version 2.0.0'
            )

    def run(self):
        _check_publish_dependencies()

        try:
            self.print_console('Removing previous builds...')
            rmtree(os.path.join(current_dir, 'dist'))
        except OSError:
            pass

        self.print_console('Tagging version v{}...'.format(self.version))
        if os.system('git tag v{}'.format(self.version)) != 0:
            raise SystemExit('Error: git tag failed. Does v{} already exist?'.format(self.version))

        self.print_console('Building Source and Wheel distribution...')
        _run_or_fail(
            '{} -m build'.format(sys.executable),
            'Build failed. Fix the errors above before publishing.'
        )

        self.print_console('Uploading the package to PyPI via Twine...')
        _run_or_fail(
            'twine upload dist/*',
            'Twine upload to PyPI failed.'
        )

        self.print_console('Pushing git tags...')
        os.system('git push --tags')

        sys.exit()


class UploadCommandTest(Command):
    """Build, tag, and publish liscal package to Test PyPI."""

    description = 'Build and publish liscal package to Test PyPI.'
    user_options = [
        ('version=', 'V', 'Version to publish (e.g. 2.0.0rc1). Required.'),
    ]

    @staticmethod
    def print_console(s):
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        self.version = None

    def finalize_options(self):
        if self.version is None:
            raise SystemExit(
                'Error: --version is required.\n'
                'Usage: python setup.py testpypi --version 2.0.0rc1'
            )

    def run(self):
        _check_publish_dependencies()

        try:
            self.print_console('Removing previous builds...')
            rmtree(os.path.join(current_dir, 'dist'))
        except OSError:
            pass

        self.print_console('Creating temporary local tag v{}...'.format(self.version))
        if os.system('git tag v{}'.format(self.version)) != 0:
            raise SystemExit('Error: git tag failed. Does v{} already exist?'.format(self.version))

        try:
            self.print_console('Building Source and Wheel distribution...')
            _run_or_fail(
                '{} -m build'.format(sys.executable),
                'Build failed. Fix the errors above before publishing.'
            )

            self.print_console('Uploading the package to Test PyPI via Twine...')
            _run_or_fail(
                'twine upload --repository testpypi dist/*',
                'Twine upload to Test PyPI failed.'
            )
        finally:
            self.print_console('Removing temporary tag v{}...'.format(self.version))
            os.system('git tag -d v{}'.format(self.version))

        sys.exit()


setup(
    scripts=[
        *[os.path.join('bin', f) for f in os.listdir('bin')
          if os.path.isfile(os.path.join('bin', f)) and f.endswith('.py')],
        os.path.join('integration', 'benchmark.py'),
        os.path.join('integration', 'test_synthetic.py'),
    ],
    cmdclass={
        'upload': UploadCommand,
        'publish': UploadCommand,
        'testpypi': UploadCommandTest,
    },
)
