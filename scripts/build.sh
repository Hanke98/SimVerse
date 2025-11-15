#!/bin/bash
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

# cmake --build build --config Release --target all -j16
cmake --build build --config Release --target TestInertial -j16
# cmake --build build --config Debug --target all -j16
# cmake --build build --config RelWithDebInfo --target all -j16
