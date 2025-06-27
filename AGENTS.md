# Codex Agents

This project is intended to be a header-only C++20 library that provides a high-performance RPC framework for AI applications. The library is designed to be used with Boost.Asio for networking and includes features like memory management, serialization, and GPU integration.

## Guidelines for Codex Agents

- Do not add any git workflows or hooks at this time. CI/CD is managed separate of the source code repository.
- Do not make any changes to the `utils.hpp` file.
- All code is formatted with `clang-format` using `.clang-format` in the repo root. Use `scripts/format.sh` before committing.
- Our focus is a minimum viable product for the microservice, and a thin mock controller that can validate the microservice's functionality.
- We need to run some complex test cases with multiple inputs and outputs.
- We currently have a `simple_cellloop.hpp` that listens to and response to zmq messages, and a `simple_engine.hpp` that needs to be wired in together.