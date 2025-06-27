#!/usr/bin/env bash
set -euo pipefail

find src tests include \( -name '*.hpp' -o -name '*.cpp' \) -print0 \
    | xargs -0 clang-format -i
