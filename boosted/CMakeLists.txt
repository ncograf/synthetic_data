# Specify minimal cmake version for this project
cmake_minimum_required(VERSION 3.28)

# Project name and version
project(boosted_stats VERSION 1.0.0 LANGUAGES CXX)

# include the eigen library
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(Boost COMPONENTS python numpy3 REQUIRED)

if (${Python3_FOUND})
    message ("Found Python " ${Python3_VERSION} " in " ${Python3_EXECUTABLE})
else()
    message ("Couldn’t find Python3!")
endif()

if (${Python3_Development_FOUND})
    message ("Found Python3 include dirs: " ${Python3_INCLUDE_DIRS})
else()
    message ("Couldn’t find Python3 Development!")
endif()

# Without this, any build libraries automatically have names "lib{x}.so"
set(CMAKE_SHARED_MODULE_PREFIX "")
set(CMAKE_STATIC_LIBRARY_PREFIX "")

include_directories(SYSTEM eigen-3.4.0)
include_directories(${PYTHON3_INCLUDE_DIRS})

# create file for LSP (make sure vscode finds eigen)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Make sure cmake finds the code section
add_subdirectory(src)

