#!/usr/bin/env python
import imp
import io
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup  # noqa: F811


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'spectral_dagger', 'version.py'))
description = ("Tools for experimenting with reinforcement learning, "
               "sequence learning and imitation learning.")
long_description = read('README.md')

setup(
    name="spectral_dagger",
    version=version_module.version,
    author="Eric Crawford",
    author_email="eric.crawford@mail.mcgill.ca",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/e2crawfo/spectral_dagger",
    license="See LICENSE.rst",
    description=description,
    long_description=long_description,
    # Without this, `setup.py install` fails to install NumPy.
    # See https://github.com/nengo/nengo/issues/508 for details.
    setup_requires=[
        "numpy>=1.7",
    ],
    install_requires=[
        "scipy",
        "sklearn",
        "pandas",
        "matplotlib",
        "dill",
        "seaborn",
        "scikits.bootstrap",
        "rdp",
        "gensim",
        "pytest",
        "hmmlearn",
        "future",
        "six",
    ],
    entry_points={
        'console_scripts': [
            'sdagger = spectral_dagger.main:main',
            'sd-parallel = spectral_dagger.utils.parallel:_sd_parallel',
            'sd-experiment = spectral_dagger.utils.parallel:_sd_experiment',
            'sd-submit = spectral_dagger.utils.submit_job:_submit_job',
        ]
    },
    zip_safe=False,
)
