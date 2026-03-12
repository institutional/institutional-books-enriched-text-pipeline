"""
setup.py - Build configuration for the simhash C++ extension module.

Build the extension:
    cd extensions/simhash_cpp
    python setup.py build_ext --inplace

Or install via pip:
    pip install ./extensions/simhash_cpp
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "1.0.0"

ext_modules = [
    Pybind11Extension(
        "_simhash_cpp",
        sources=["simhash_bindings.cpp", "simhash_impl.cpp"],
        include_dirs=[],
        define_macros=[("VERSION_INFO", __version__)],
        extra_compile_args=[
            "-O3",
            "-march=native",  # Enable native CPU features (POPCNT)
        ],
        cxx_std=17,
    ),
]

setup(
    name="simhash_cpp",
    version=__version__,
    author="David Lowry-Duda",
    author_email="dlowry@law.harvard.edu",
    description="High-performance simhash implementation in C++",
    long_description="Provides 128-bit simhash computation using MurmurHash3 with character ngrams",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["pybind11>=2.10.0"],
)
