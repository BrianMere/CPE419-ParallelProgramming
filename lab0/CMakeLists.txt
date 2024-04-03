cmake_minimum_required(VERSION 3.5)

# Name of the project
project(lab0 CXX)

# Use glob to get the list of all source files.
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.c")

# We don't really need to include header and resource files to build, but it's
# nice to have them show up in IDEs.
include_directories("src/include")
file(GLOB_RECURSE HEADERS "src/include/*.h")

# Set the executable.
add_executable(${CMAKE_PROJECT_NAME} ${SOURCES} ${HEADERS}) # ${HEADERS}
# and the path where it goes "usually bin"
set(EXECUTABLE_OUTPUT_PATH "../bin")

# OS specific options and libraries
if(WIN32)
  # c++0x is enabled by default.
  # -Wall produces way too many warnings.
  # -pedantic is not supported.
  # Disable warning 4996.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
else()
  # Enable all pedantic warnings.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall -pedantic")
endif()

