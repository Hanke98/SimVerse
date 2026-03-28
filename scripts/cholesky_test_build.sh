#!/bin/bash
set -euo pipefail

# Usage:
#   ./scripts/cholesky_test_build.sh [rel|reldeb|relwithdebug|debug] [run]
build_arg="${1:-rel}"
run_arg="${2:-}"

if [ "$#" -gt 2 ]; then
  echo "Usage: $0 [build_arg] [run]"
  exit 1
fi

if [ -n "$run_arg" ] && [ "$run_arg" != "run" ]; then
  echo "Unknown second argument: $run_arg"
  echo "Usage: $0 [build_arg] [run]"
  exit 1
fi

PERIDYNO_C_COMPILER="${PERIDYNO_C_COMPILER:-gcc}"
PERIDYNO_CXX_COMPILER="${PERIDYNO_CXX_COMPILER:-g++}"

target="Test_Cholesky"
build_dir="build"

cmake -B "$build_dir" -S . -G "Ninja Multi-Config" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER="$PERIDYNO_CXX_COMPILER" \
  -DCMAKE_C_COMPILER="$PERIDYNO_C_COMPILER" \
  -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized" \
  -DCMAKE_PREFIX_PATH="${PERIDYNO_CMAKE_PREFIX_PATH:-}" \
  -DPERIDYNO_TESTS=ON \
  -DPERIDYNO_EXAMPLE=OFF \
  -DPERIDYNO_LIBRARY_FRAMEWORK=ON \
  -DPERIDYNO_LIBRARY_SIMNODE=ON

case "$build_arg" in
  rel)
    config="Release"
    ;;
  reldeb|relwithdebug)
    config="RelWithDebInfo"
    ;;
  debug)
    config="Debug"
    ;;
  *)
    echo "Unknown build option: $build_arg"
    echo "Supported build_arg: rel | reldeb | relwithdebug | debug"
    exit 1
    ;;
esac

cmake --build "$build_dir" --config "$config" -j32 --target "$target"

if [ "$run_arg" != "run" ]; then
  exit 0
fi

log_folder="./logs"
mkdir -p "$log_folder"

exe_path="./${build_dir}/bin/Release/${config}/${target}"
if [ ! -x "$exe_path" ]; then
  fallback_path="./${build_dir}/${target}"
  if [ -x "$fallback_path" ]; then
    exe_path="$fallback_path"
  else
    echo "Executable not found: $exe_path"
    exit 1
  fi
fi

"$exe_path" > "$log_folder/log" 2> "$log_folder/err"
