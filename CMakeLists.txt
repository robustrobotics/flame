cmake_minimum_required(VERSION 2.8)

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

project(flame)
set(${PROJECT_NAME}_VERSION 1.0.0)

# Options
option(BUILD_DOCUMENTATION "Use Doxygen to create the HTML based API documentation" OFF)
option(BUILD_TESTS "Use GTest to build and test libraries" ON)

option(WITH_COVERAGE "Compile with code coverage" OFF)
if (WITH_COVERAGE)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fprofile-arcs -ftest-coverage")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
endif ()

add_definitions("-std=c++11")

# Add package setup file.
include(cmake/setup.cmake)

# Locate external dependencies
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs)
find_package(Boost 1.54 REQUIRED COMPONENTS system filesystem graph)
find_package(Sophus REQUIRED)

find_package(PkgConfig REQUIRED) # For projects with pkg-config files.
pkg_check_modules(EIGEN3 REQUIRED eigen3>=3.2)

# Project includes
include_directories(SYSTEM src
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS})

# Build the library
add_subdirectory(src)

# Configure the library
# Configuration method 1: pkg_config
include(cmake/pkg_config.cmake)
install_pkg_config_file(${PROJECT_NAME}
  DESCRIPTION "A lightweight method for dense online monocular depth estimation."
  LIBS ${PROJECT_NAME}
  VERSION ${${PROJECT_NAME}_VERSION}
  REQUIRES opencv eigen3>=3.2
  CFLAGS "-std=c++11")

# Configuration method 2: pure cmake, no import/export
configure_file(${PROJECT_SOURCE_DIR}/cmake/templates/${PROJECT_NAME}Config.cmake.in
  "${PROJECT_BINARY_DIR}/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake" @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/cmake/templates/${PROJECT_NAME}ConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)
install(FILES 
  ${PROJECT_BINARY_DIR}/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake 
  ${PROJECT_BINARY_DIR}/lib/cmake/${PROJECT_NAME}/${PROJECT_NAME}ConfigVersion.cmake 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME})

# If the BUILD_TESTS option is set, a Google test binary will be compiled.
if (BUILD_TESTS)
  add_subdirectory(./test)
endif()

# Build the doxygen documentation.
if (BUILD_DOCUMENTATION)
  find_package(Doxygen)

  if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR
      "Doxygen is needed to build the documentation.")
  endif()

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile @ONLY)
  add_custom_target(doc ALL
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc
    COMMENT "Generating API documentation with Doxygen" VERBATIM)

  install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/html DESTINATION doc/${PROJECT_NAME})
  install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/latex DESTINATION doc/${PROJECT_NAME})
endif()

