"""
Setup script for Psyne Python bindings
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import os
import sys

# Get the parent directory (Psyne root)
psyne_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "psyne",
        [
            "psyne_bindings.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to Psyne headers
            os.path.join(psyne_root, "include"),
            os.path.join(psyne_root, "src", "utils"),
            # Boost headers (for ASIO)
            os.path.expanduser("~/boost"),
        ],
        libraries=["psyne"],
        library_dirs=[
            os.path.join(psyne_root, "build"),  # Where libpsyne.so/dylib is built
        ],
        language='c++',
        cxx_std=20,
        define_macros=[
            ("VERSION_INFO", '"{dev}"'),
        ],
    ),
]

setup(
    name="psyne",
    version="1.2.0",
    author="Psyne Contributors",
    author_email="",
    description="High-performance zero-copy messaging library for Python",
    long_description="""
Psyne is a high-performance, zero-copy messaging library optimized for AI/ML applications.
It provides efficient inter-process communication with support for:

- Multiple transport types (IPC, TCP, Unix sockets, UDP multicast)
- Zero-copy message passing
- Compression support (LZ4, Zstd, Snappy)
- NumPy integration for efficient data transfer
- Thread-safe channels with different synchronization modes
- Comprehensive metrics and debugging utilities

This Python package provides bindings to the C++ Psyne library, enabling
Python applications to leverage Psyne's performance benefits.
""",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.16.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
)