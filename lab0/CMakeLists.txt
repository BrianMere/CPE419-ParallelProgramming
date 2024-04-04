cmake_minimum_required(VERSION 2.8)

# Name of the project
project(lab0)

# Use glob to get the list of all source files.
file(GLOB_RECURSE SOURCES_CPP "src/*.cpp")
file(GLOB_RECURSE SOURCES_C "src/*.c")

set(SOURCES ${SOURCES_CPP} ${SOURCES_C})

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Set the include directories.
include_directories("./include")

# Set compiler flags and linker flags
SET(COMPILE_FLAGS "-Wall -pedantic -fpermissive ")
SET(LINK_FLAGS    "-lrt")

# Set the executable.
add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

# Set the output directories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# OS specific options and libraries
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${LINK_FLAGS}")

