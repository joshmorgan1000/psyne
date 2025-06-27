# Psyne Source Directory

This directory contains the implementation files for the Psyne library.

## Structure

- `psyne.cpp` - Main library implementation file (minimal, as Psyne is primarily header-only)
- `utils/` - Utility headers and implementations
  - `utils.hpp` - General utility functions
  - `xxhash32.h` - Fast hash function for message integrity
  - `vthread.hpp` - Thread utilities
  - `nlohmann/` - JSON library (for future configuration support)

## Notes

Psyne is designed as a header-only library for maximum performance and ease of integration. The `psyne.cpp` file exists primarily to:
1. Satisfy build systems that expect a source file
2. Provide version information functions
3. Hold any future non-template implementations

Most of the actual implementation is in the header files under `include/psyne/`.