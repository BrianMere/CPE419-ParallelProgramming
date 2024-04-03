cmake_minimum_required(VERSION 3.5)

# Name of the project
project(lab0 CXX)

# Use glob to get the list of all source files.
file(GLOB_RECURSE SOURCES "src/*.cpp")


# Set the include directories.
include_directories("./include")

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -pedantic")
endif()
