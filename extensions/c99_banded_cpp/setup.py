"""
setup.py - Build configuration for the C99 chunking extension.

Build the extension:
    cd extensions/c99_banded_cpp
    python setup.py build_ext --inplace

Or install via pip:
    pip install ./extensions/c99_banded_cpp

Institutional Books - Enriched Text - 2026
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "1.0.0"

ext_modules = [
    Pybind11Extension(
        "_c99_cpp",
        sources=["c99_banded.cpp"],
        include_dirs=[],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=[
            "-O3",
            "-march=native",
        ],
        cxx_std=17,
    ),
]

setup(
    name="c99_cpp",
    version=__version__,
    author="David Lowry-Duda",
    author_email="dlowry@law.harvard.edu",
    description="C99 topic segmentation algorithm in C++",
    long_description="Implements C99 banded similarity segmentation for topic detection",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["pybind11>=2.10.0", "numpy>=1.20.0"],
)
