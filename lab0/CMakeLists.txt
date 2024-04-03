cmake_minimum_required(VERSION 2.8)

# Name of the project
project(lab0)

# Use glob to get the list of all source files.
file(GLOB_RECURSE SOURCES_CPP "src/*.cpp")
file(GLOB_RECURSE SOURCES_C "src/*.c")

set(SOURCES ${SOURCES_CPP} ${SOURCES_C})

# Set the include directories.
include_directories("./include")

# Set compiler flags
set(COMMON_FLAGS "-Wall -pedantic -fpermissive")
set(CMAKE_C_FLAGS "${COMMON_FLAGS}")

# Set the executable.
add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

# Set the output directories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# OS specific options and libraries
if(WIN32)
  # Disable warning 4996.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
else()
  # Enable all pedantic warnings.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x ${COMMON_FLAGS}")
endif()
