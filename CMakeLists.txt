cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(ngam C CXX)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type defaults to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
# -march=native is required so that ngam and its dependencies (DisORT++, FastChem)
# agree on SIMD alignment for Eigen types
add_compile_options(-Wall -pedantic -march=native)

# Place executable in the source directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# When building Python bindings, all libraries (including fetched dependencies)
# must be compiled with position-independent code
if(BUILD_PYTHON_BINDINGS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()


# Dependencies
include(FetchContent)

FetchContent_Declare(
  disortpp
  GIT_REPOSITORY https://github.com/NewStrangeWorlds/DisORT
  GIT_TAG        v1.1
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(disortpp)

set(FASTCHEM_ONLY_LIBRARY ON)
FetchContent_Declare(
  fastchem
  GIT_REPOSITORY https://github.com/newstrangeworlds/fastchem
  GIT_TAG        cf931e730aaa88315e913562b4f468cc0fd39712
)
FetchContent_MakeAvailable(fastchem)

find_package(OpenMP REQUIRED C CXX)

if(BUILD_PYTHON_BINDINGS)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(pybind11)
endif()


# Library sources (everything except main.cpp)
set(LIB_SRC
  src/additional/aux_functions.cpp
  src/additional/thermodynamic_data.cpp
  src/config/global_config.cpp
  src/spectral_grid/convert.cpp
  src/spectral_grid/spectral_grid.cpp
  src/spectral_grid/spectral_grid_interpolate.cpp
  src/transport_coeff/cross_section_file.cpp
  src/transport_coeff/opacity_species.cpp
  src/transport_coeff/sampled_data.cpp
  src/transport_coeff/species_rayleigh_cross_sections.cpp
  src/transport_coeff/transport_coeff.cpp
  src/transport_coeff/h_m.cpp
  src/radiative_transfer/discrete_ordinate.cpp
  src/radiative_transfer/short_characteristics.cpp
  src/atmosphere/atmosphere.cpp
  src/chemistry/isoprofile_chemistry.cpp
  src/chemistry/fastchem_chemistry.cpp
  src/temperature/milne_solution_temperature.cpp
  src/temperature/time_stepping_temperature.cpp
  src/convection/dry_adiabatic.cpp
)

# Static library
add_library(ngam_lib STATIC ${LIB_SRC})
target_link_libraries(ngam_lib PUBLIC fastchem_lib OpenMP::OpenMP_CXX disortpp)
set_target_properties(ngam_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Main executable
add_executable(ngam src/main.cpp)
target_link_libraries(ngam PRIVATE ngam_lib)

# Python bindings (optional)
if(BUILD_PYTHON_BINDINGS)
  pybind11_add_module(pyngam src/python/bindings.cpp)
  target_link_libraries(pyngam PRIVATE ngam_lib)
  install(TARGETS pyngam LIBRARY DESTINATION .)
endif()
