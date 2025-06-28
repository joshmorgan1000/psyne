# Agent instructions for Psyne

This project is intended to be a low-level C++20 RPC library that is optimized for AI/ML applications.

Code may not be correct. When in doubt, refer to the [README.md](README.md) document for the latest architecture and design decisions.

## Guidelines for Codex Agents

- Do not add any git workflows or hooks at this time. CI/CD is managed separate of the source code repository.
- Do not make any changes to the `utils.hpp` file.
- Header-only classes are preferred unless unavoiable.
- All code must be unit tested.
- All code must be documented with Doxygen comments in javadoc style.
- We would like to minimize the dependencies on third-party libraries.
- Keep all code in the `src/` directory, but make sure all necessary headers are included in the `include/psyne/psyne.hpp` file.
- All code is formatted with `clang-format` using `.clang-format` in the repo root. Use `scripts/format.sh` before committing.