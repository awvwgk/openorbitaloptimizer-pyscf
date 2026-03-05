cmake_minimum_required(VERSION 3.17)
project(openorbitaloptimizer_python LANGUAGES CXX)

find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

find_package(Armadillo REQUIRED CONFIG)

# Try to find an installed OpenOrbitalOptimizer; fetch from GitHub if not found
find_package(OpenOrbitalOptimizer QUIET CONFIG)
if(NOT OpenOrbitalOptimizer_FOUND)
  message(STATUS "OpenOrbitalOptimizer not found – fetching from GitHub")
  include(FetchContent)
  FetchContent_Declare(
    OpenOrbitalOptimizer
    GIT_REPOSITORY https://github.com/susilehtola/openorbitaloptimizer.git
    GIT_TAG master
  )
  FetchContent_MakeAvailable(OpenOrbitalOptimizer)
endif()

# Build the nanobind module
nanobind_add_module(_core STABLE_ABI src/_core.cpp)

target_link_libraries(_core PRIVATE
  OpenOrbitalOptimizer::OpenOrbitalOptimizer
  armadillo
)
target_compile_features(_core PRIVATE cxx_std_17)

install(TARGETS _core LIBRARY DESTINATION openorbitaloptimizer)
