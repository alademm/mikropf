﻿cmake_minimum_required (VERSION 3.8)
project ("MikroPF")
add_library(${PROJECT_NAME} "MikroPF.cpp" "MikroPF.h")
add_executable (example "example.cpp")
target_link_libraries(example ${PROJECT_NAME})