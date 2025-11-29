#!/bin/bash

# Default optimization option
run_arg="rel"

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
  run_arg=$1
elif [ "$#" -gt 1 ]; then
  echo "Usage: $0 [option]"
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
-DCMAKE_PREFIX_PATH=$PERIDYNO_CMAKE_PREFIX_PATH
# --fresh

target=RobotArm_Impulse
target=TestInertial
case $run_arg in
rel)
  cmake --build build --config Release -j32 --target $target
  ;;
reldeb)
  cmake --build build --config RelWithDebInfo -j32 --target $target
  ;;
debug)
  cmake --build build --config Debug -j32 --target $target
  ;;
*)
  echo "Unknown option: $1"
  exit 1
  ;;
esac
