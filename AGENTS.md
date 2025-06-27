# Codex Agents

This project is intended to be a header-only C++20 library that provides a high-performance RPC framework for AI applications. For more information see the [README](README.md) and the [PHILOSOPHY](PHILOSOPHY.md) documents.

## Guidelines for Codex Agents

- Do not add any git workflows or hooks at this time. CI/CD is managed separate of the source code repository.
- Do not make any changes to the `utils.hpp` file.
- All code is formatted with `clang-format` using `.clang-format` in the repo root. Use `scripts/format.sh` before committing.