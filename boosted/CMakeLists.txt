# Specify minimal cmake version for this project
cmake_minimum_required(VERSION 3.24)

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

# GoogleTests
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# create file for LSP (make sure vscode finds eigen)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

# Make sure cmake finds the code section
add_subdirectory(src)

