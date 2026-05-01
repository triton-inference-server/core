# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.31.8)

add_subdirectory(tritonserver)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/TRITON_VERSION ${TRITON_VERSION})
configure_file(../LICENSE LICENSE.txt COPYONLY)
configure_file(setup.py setup.py @ONLY)
configure_file(../pyproject.toml pyproject.toml COPYONLY)
file(COPY test/  DESTINATION ./test/.)

set(WHEEL_DEPENDS
      ${CMAKE_CURRENT_BINARY_DIR}/TRITON_VERSION
      ${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt
      ${CMAKE_CURRENT_BINARY_DIR}/setup.py
      ${CMAKE_CURRENT_BINARY_DIR}/tritonserver
      python-bindings
)

set(wheel_stamp_file "stamp.whl")

add_custom_command(
  OUTPUT "${wheel_stamp_file}"
  COMMAND python3
  ARGS
    "${CMAKE_CURRENT_SOURCE_DIR}/build_wheel.py"
    --dest-dir "${CMAKE_CURRENT_BINARY_DIR}/generic"
    --binding-path $<TARGET_FILE:python-bindings>
  DEPENDS ${WHEEL_DEPENDS}
)

add_custom_target(
  generic-server-wheel ALL
  DEPENDS
    "${wheel_stamp_file}"
)


# Wheel
set(WHEEL_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generic/wheel/dist/")
install(
  DIRECTORY
  ${WHEEL_OUT_DIR}
  DESTINATION "${CMAKE_INSTALL_PREFIX}/python"
)


# Tests
set(TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/test")
install(
  DIRECTORY
  ${TEST_DIR}
  DESTINATION "${CMAKE_INSTALL_PREFIX}/python"
)
