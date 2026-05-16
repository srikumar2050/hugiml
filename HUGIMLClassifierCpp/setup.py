"""
setup.py — builds the _hugiml_core pybind11 C++ extension.

Install for development (compiles the .so / .pyd in-place):
    pip install -e .

Build a redistributable wheel:
    pip install build
    python -m build --wheel
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class BuildExt(build_ext):
    """Add platform-specific optimisation flags."""

    def build_extensions(self):
        ct   = self.compiler.compiler_type
        opts = []

        if ct == 'unix':
            opts += ['-O3', '-march=native', '-fvisibility=hidden', '-std=c++17']
            if sys.platform != 'darwin':
                opts.append('-flto')
        elif ct == 'msvc':
            opts += ['/O2', '/std:c++17']

        for ext in self.extensions:
            ext.extra_compile_args = opts
            if ct == 'unix' and sys.platform != 'darwin' and '-flto' in opts:
                ext.extra_link_args = ['-flto']

        super().build_extensions()


ext = Extension(
    '_hugiml_core',
    sources=['src/hugiml_core.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
)

setup(
    name='hugiml-native',
    version='1.0.0',
    author='Srikumar Krishnamoorthy',
    description='HUG-IML Classifier — C++ accelerated, scikit-learn compatible',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    license='GPL-3.0',
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.22',
        'scipy>=1.8',
        'scikit-learn>=1.0',
        'pandas>=1.4',
        'pybind11>=2.10',
    ],
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExt},
    py_modules=['HUGIMLClassifierNative'],
    zip_safe=False,
)
