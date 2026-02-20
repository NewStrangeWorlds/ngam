
# to build with cmake
# create a build directory and move into it
# $ mkdir build
# $ cd build
# generate the makefile (to do only ones, if we don't add files or change makefiles)
# don't forget the two points at the end of the command '..'.
# It runs cmake in the 'build' directory
# but with the data from the '..' directory.
# the script should autodetect the CUDA architecture, when run alone
# $ cmake ..
# You can specify the architecture with the SM variable:
# $ cmake -DSM=30 ..

cmake_minimum_required (VERSION 3.10 FATAL_ERROR)

OPTION(USE_PYTHON "Compile BeAR's Python Wrapper" OFF)


project(ngam C CXX)


#some C++/C flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -pedantic -MMD -march=native")
set(CMAKE_C_FLAGS "-Wall -pedantic -MMD -march=native")


set(CMAKE_BUILD_TYPE Release)

  
  
#output directories
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})


include(FetchContent)
FetchContent_Declare(
    disortpp
    GIT_REPOSITORY https://github.com/NewStrangeWorlds/DisORT
    GIT_TAG        v1.0.3
    GIT_SHALLOW    TRUE
)

FetchContent_GetProperties(disortpp)
if(NOT disortpp_POPULATED)
    FetchContent_Populate(disortpp)
    add_subdirectory(${disortpp_SOURCE_DIR} ${disortpp_BINARY_DIR})
endif()


#download FastChem 3, if necessary
FetchContent_Declare(
  fastchem
  GIT_REPOSITORY https://github.com/newstrangeworlds/fastchem
  GIT_TAG        cf931e730aaa88315e913562b4f468cc0fd39712
)

FetchContent_GetProperties(fastchem)
set(FASTCHEM_ONLY_LIBRARY ON)

if (NOT fastchem_POPULATED)
  FetchContent_Populate(fastchem)
  add_subdirectory(${fastchem_SOURCE_DIR} ${fastchem_BINARY_DIR})
endif()


#define the source files 
set(SRC
  src/main.cpp
  src/additional/aux_functions.cpp
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
  src/temperature/milne_solution_temperature.cpp)


#check for OpenMP
find_package(OpenMP REQUIRED C CXX)


# build application for BeAR
add_executable(ngam ${SRC})
#target_include_directories(ngam)

target_include_directories(ngam
    PUBLIC disortpp
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:src>
)

#link files for the final BeAR code
#target_link_libraries(bear PUBLIC fastchem_lib bear_cuda multinest_shared OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES})
target_link_libraries(ngam PUBLIC fastchem_lib OpenMP::OpenMP_CXX disortpp)
