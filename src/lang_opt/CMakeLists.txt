cmake_minimum_required(VERSION 3.15)
project(lang_opt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force Python 3.13 - use the one from environment variable or passed to CMake
if(DEFINED ENV{Python_ROOT_DIR})
    set(Python_ROOT_DIR $ENV{Python_ROOT_DIR})
endif()

set(Python_FIND_STRATEGY LOCATION)
set(Python_FIND_VIRTUALENV FIRST)

# Find Python and pybind11 - require 3.13
find_package(Python 3.13...<3.14 COMPONENTS Interpreter Development REQUIRED)

# Log which Python was found
message(STATUS "Found Python ${Python_VERSION}: ${Python_EXECUTABLE}")
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    # If pybind11 is not found, try to use it as a submodule or download it
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Create the Python module
pybind11_add_module(lang_opt_native
    src/lexer.cpp
    src/parser.cpp
    src/bindings.cpp
)

# Set compile options for optimization
target_compile_options(lang_opt_native PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -flto -ffast-math>
    $<$<CONFIG:Debug>:-g -O0>
)

# Enable link-time optimization for Release builds
set_target_properties(lang_opt_native PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
)

# Installation
install(TARGETS lang_opt_native DESTINATION .)

