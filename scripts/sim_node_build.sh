#!/bin/bash

# Default building option
build_arg="rel"
run_arg=""

# Parse arguments: [build_arg] [run]
if [ "$#" -ge 1 ]; then
  build_arg=$1
fi

if [ "$#" -ge 2 ]; then
  run_arg=$2
fi

if [ "$#" -gt 2 ]; then
  echo "Usage: $0 [build_arg] [run]"
  exit 1
fi

if [ -n "$run_arg" ] && [ "$run_arg" != "run" ]; then
  echo "Unknown second argument: $run_arg"
  echo "Usage: $0 [build_arg] [run]"
  exit 1
fi

if [ -z "$PERIDYNO_C_COMPILER" ]; then
  PERIDYNO_C_COMPILER=gcc
fi

if [ -z "$PERIDYNO_CXX_COMPILER" ]; then
  PERIDYNO_CXX_COMPILER=g++
fi

cmake -B build -S . -G "Ninja Multi-Config" \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DPERIDYNO_LIBRARY_DUALPARTICLESYSTEM=OFF \
-DCMAKE_CXX_COMPILER=$PERIDYNO_CXX_COMPILER \
-DCMAKE_C_COMPILER=$PERIDYNO_C_COMPILER \
-DPERIDYNO_PLUGIN_FBX=ON \
-DPERIDYNO_QT_GUI=OFF \
-DPERIDYNO_PLUGIN_MUJOCO=OFF \
-DPERIDYNO_USE_SYSTEM_GLFW=$PERIDYNO_USE_SYSTEM_GLFW \
-DCMAKE_PREFIX_PATH=$PERIDYNO_CMAKE_PREFIX_PATH \
# --fresh

# target
target=SimNodeInit

case $build_arg in
rel)
  cmake --build build --config Release -j32 --target $target
  ;;
reldeb)
  cmake --build build --config RelWithDebInfo -j32 --target $target
  ;;
relwithdebug)
  cmake --build build --config RelWithDebInfo -j32 --target $target
  ;;
debug)
  cmake --build build --config Debug -j32 --target $target
  ;;
*)
  echo "Unknown build option: $build_arg"
  echo "Supported build_arg: rel | reldeb | relwithdebug | debug"
  exit 1
  ;;
esac

if [ "$run_arg" != "run" ]; then
  exit 0
fi

log_folder="./logs"
mkdir -p "$log_folder"

bin_folder=""
case $build_arg in
debug)
  bin_folder="./build/bin/Release/Debug"
  ;;
reldeb|relwithdebug)
  bin_folder="./build/bin/Release/RelWithDebInfo"
  ;;
*)
  bin_folder="./build/bin/Release/Release"
  ;;
esac

exe_path="$bin_folder/$target"
if [ ! -x "$exe_path" ]; then
  fallback_path="./build/$target"
  if [ -x "$fallback_path" ]; then
    exe_path="$fallback_path"
  else
    echo "Executable not found: $exe_path"
    exit 1
  fi
fi

"$exe_path" > "$log_folder/log" 2> "$log_folder/err"

