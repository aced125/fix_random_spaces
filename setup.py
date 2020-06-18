#!/usr/bin/env python
# fmt: off
"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

test_requirements = ['pytest>=3', ]

setup(
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    license="MIT license",
    include_package_data=True,
    keywords='fix_random_spaces',
    packages=find_packages(include=['fix_random_spaces', 'fix_random_spaces.*']),
    test_suite='tests',
    version='0.1.0',
    zip_safe=False,
)
