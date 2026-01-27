
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
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -pedantic -MMD")
set(CMAKE_C_FLAGS "-Wall -pedantic -MMD")


set(CMAKE_BUILD_TYPE Release)

  
  
#output directories
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})




#define the source files 
set(SRC
  src/main.cpp
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
  src/radiative_transfer/cdisort_src/cdisort.c
  src/radiative_transfer/cdisort_src/locate.c)


# set(SRC_PYTHON_WRAPPER
#   python/src/bear_python_wrapper.cpp)


#check for OpenMP
find_package(OpenMP REQUIRED C CXX)



# include(FetchContent)
#
# #download the Boost Math library if necessary
# FetchContent_Declare(
#   boost_math
#   GIT_REPOSITORY https://github.com/boostorg/math
#   GIT_TAG        ed01dae24893bb69c02c6d599acc74bdb8f46bda
# )
#
# FetchContent_GetProperties(boost_math)
#
# if (NOT boost_math_POPULATED)
#   FetchContent_Populate(boost_math)
#   add_subdirectory(${boost_math_SOURCE_DIR} ${boost_math_BINARY_DIR})
# endif()



#download FastChem 3, if necessary
# FetchContent_Declare(
#   fastchem
#   GIT_REPOSITORY https://github.com/newstrangeworlds/fastchem
#   GIT_TAG        0979ebe637de51241c77e73071b0a3da5a16c8f2
# )
#
# FetchContent_GetProperties(fastchem)
# set(FASTCHEM_ONLY_LIBRARY ON)
#
# if (NOT fastchem_POPULATED)
#   FetchContent_Populate(fastchem)
#   add_subdirectory(${fastchem_SOURCE_DIR} ${fastchem_BINARY_DIR})
# endif()


# if (USE_PYTHON MATCHES ON)
#     #find the python version
#     find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#
#     set(PYBIND11_FINDPYTHON ON)
#
#     include(FetchContent)
#     FetchContent_Declare(
#         pybind11
#         GIT_REPOSITORY https://github.com/pybind/pybind11
#         GIT_TAG        7c33cdc2d39c7b99a122579f53bc94c8eb3332ff
#     )
#
#     FetchContent_GetProperties(pybind11)
#     if(NOT pybind11_POPULATED)
#         FetchContent_Populate(pybind11)
#         add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
#     endif()
#
#
#
#   #compilation target for BeAR's Python library
#   add_library(bear_for_py ${SRC})
#   target_include_directories(bear_for_py PUBLIC ${boost_math_SOURCE_DIR}/include/)
#
#
#   target_compile_options(bear_for_py PRIVATE "${OpenMP_CXX_FLAGS}" PRIVATE "-O3" PRIVATE "-fPIC")
#
#
#   #compilation target for BeAR's Python wrapper
#   pybind11_add_module(pybear ${SRC_PYTHON_WRAPPER})
#
#   set_target_properties(pybear PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/lib")
#
#   target_link_libraries(pybear PUBLIC fastchem_lib_for_py bear_for_py multinest_shared OpenMP::OpenMP_CXX pybind11::module ${LAPACK_LIBRARIES})
# endif()



# build application for BeAR
add_executable(ngam ${SRC})
#target_include_directories(ngam)


#link files for the final BeAR code
#target_link_libraries(bear PUBLIC fastchem_lib bear_cuda multinest_shared OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES})
target_link_libraries(ngam PUBLIC OpenMP::OpenMP_CXX)
