"""
setup.py - Build configuration including C++ extensions

This setup.py enables building C++ extensions when the package is installed.
Follow the README for standard building instructions.

The extensions provide significant performance improvements:

- extensions.built._simhash_cpp: 128-bit simhash using MurmurHash3
- extensions.built._c99_cpp: C99 topic segmentation algorithm

Extensions are optional - the library falls back to pure Python implementations
when extensions are not available.

Build extensions:
    python setup.py build_ext --inplace

Install with extensions:
    pip install .
"""

import sys
from pathlib import Path

from setuptools import setup

# Check if pybind11 is available for building extensions
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    PYBIND11_AVAILABLE = True
except ImportError:
    PYBIND11_AVAILABLE = False
    Pybind11Extension = None
    build_ext = None


def get_extensions():
    """Get list of C++ extensions to build."""
    if not PYBIND11_AVAILABLE:
        print("Warning: pybind11 not available, skipping C++ extensions")
        return []

    extensions = []

    # Common compile args for performance
    extra_compile_args = ["-O3"]

    # Add -march=native on non-Windows platforms
    # (This has never been tested on a Windows platform)
    if sys.platform != "win32":
        extra_compile_args.append("-march=native")

    # Simhash extension
    simhash_sources = [
        "extensions/simhash_cpp/simhash_bindings.cpp",
        "extensions/simhash_cpp/simhash_impl.cpp",
    ]
    if all(Path(src).exists() for src in simhash_sources):
        extensions.append(
            Pybind11Extension(
                "extensions.built._simhash_cpp",
                sources=simhash_sources,
                extra_compile_args=extra_compile_args,
                cxx_std=17,
            )
        )

    # C99 chunking extension
    c99_sources = ["extensions/c99_cpp/c99_banded.cpp"]
    if all(Path(src).exists() for src in c99_sources):
        extensions.append(
            Pybind11Extension(
                "extensions.built._c99_cpp",
                sources=c99_sources,
                extra_compile_args=extra_compile_args,
                cxx_std=17,
            )
        )

    return extensions


# Only use custom cmdclass if we can build extensions
cmdclass = {}
if PYBIND11_AVAILABLE and build_ext is not None:
    cmdclass["build_ext"] = build_ext

setup(
    ext_modules=get_extensions(),
    cmdclass=cmdclass,
)
