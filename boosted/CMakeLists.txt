# Specify minimal cmake version for the synthetic data project
cmake_minimum_required(VERSION 3.24)

# Build boost_python and boost_numpy as shared libs (does not work otherwise)
option(HUNTER_BUILD_SHARED_LIBS "Build Boost libraries as shared libs" ON)

# Get hunter as "Package Manager" MUST BE BEFORE project
include("cmake/HunterGate.cmake")
HunterGate(
    URL "https://github.com/cpp-pm/hunter/archive/v0.25.6.tar.gz"
    SHA1 "69e4a05bd514bb05cb2c5aa02ce8ac420516aaf7"
)

# Project name and version
project(boosted_stats VERSION 2.0.0)

########################
# Install dependencies #
########################
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

# include python and python header files
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
# Hack to make sure Hunter installs Boostlib Python with correct headers
set(ENV{CPLUS_INCLUDE_PATH} ${Python3_INCLUDE_DIRS})

# include the boost library
hunter_add_package(Boost COMPONENTS python)
find_package(Boost CONFIG REQUIRED python311 numpy311)
set_target_properties(Boost::python311 Boost::numpy311 PROPERTIES POSITION_INDEPENDENT_CODE ON)


####################################
# Configure output and compilation #
####################################

# Without this, any build libraries automatically have names "lib{x}.so"
set(CMAKE_SHARED_MODULE_PREFIX "")
set(CMAKE_STATIC_LIBRARY_PREFIX "")

# create file for LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# make fast compilation
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

# Make sure cmake finds the code section
add_subdirectory(src)

