#!/usr/bin/env bash
set -euo pipefail

find src tests include examples benchmarks \( -name '*.hpp' -o -name '*.cpp' \) -print0 \
    | xargs -0 clang-format -i
