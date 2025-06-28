# Agent instructions for Psyne

This project is intended to be a low-level C++20 RPC library that is optimized for AI/ML applications. I believe it could be the fastest and most efficient way to transport tensors between neural network layers in existence.

## Guidelines for Codex Agents

- Header-only classes are preferred unless unavoiable.
- All code must be unit tested.
- All code must be documented with Doxygen comments in javadoc style.
- We would like to minimize the dependencies on third-party libraries.
- Keep all code in the `src/` directory, but make sure all necessary headers are included in the `include/psyne/psyne.hpp` file.
- All code is formatted with `clang-format` using `.clang-format` in the repo root. Use `scripts/format.sh` before committing.