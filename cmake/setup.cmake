#------------#
# Set custom options
#------------#
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  message(STATUS "Installing to ${CMAKE_SOURCE_DIR}/install")
  message(WARNING "CMake defaults to installing to ${CMAKE_INSTALL_PREFIX}. This is probably not what you want, and so this build system installs locally, to ${CMAKE_SOURCE_DIR}/install.  To override this behavior, set CMAKE_INSTALL_PREFIX manually.")
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "${PROJECT_NAME} install prefix" FORCE)
else()
  message(STATUS "Installing to ${CMAKE_INSTALL_PREFIX}")
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build: Debug|Release|RelWithDebInfo|MinSizeRel." FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#------------#
# Set up paths
#------------#
# set cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_INSTALL_PREFIX}/lib/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_MODULE_PATH} ${CMAKE_INSTALL_PREFIX}/lib/cmake)

# set where files should be output locally
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(INCLUDE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/include)
set(PKG_CONFIG_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/pkgconfig)

# set where files should be installed to
set(LIBRARY_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib)
set(EXECUTABLE_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)
set(INCLUDE_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/include)
set(PKG_CONFIG_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)

# add build/lib/pkgconfig to the pkg-config search path
# wrvb 2014-11-02: CMake shouldn't touch environment variables unless it has to
# set(ENV{PKG_CONFIG_PATH} ${PKG_CONFIG_INSTALL_PATH}:$ENV{PKG_CONFIG_PATH})
# set(ENV{PKG_CONFIG_PATH} ${PKG_CONFIG_OUTPUT_PATH}:$ENV{PKG_CONFIG_PATH})
# set(ENV{PKG_CONFIG_PATH} ${CMAKE_PREFIX_PATH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH})

# add include directories to the compiler include path
# include_directories(include)
# include_directories(BEFORE ${INCLUDE_OUTPUT_PATH})
# include_directories(${INCLUDE_INSTALL_PATH})

# add build/lib to the link path
link_directories(${LIBRARY_OUTPUT_PATH})
link_directories(${LIBRARY_INSTALL_PATH})

# abuse RPATH
if(${CMAKE_INSTALL_RPATH})
  set(CMAKE_INSTALL_RPATH ${LIBRARY_INSTALL_PATH}:${CMAKE_INSTALL_RPATH})
else(${CMAKE_INSTALL_RPATH})
  set(CMAKE_INSTALL_RPATH ${LIBRARY_INSTALL_PATH})
endif(${CMAKE_INSTALL_RPATH})

# for osx, which uses "install name" path rather than rpath
#set(CMAKE_INSTALL_NAME_DIR ${LIBRARY_OUTPUT_PATH})
set(CMAKE_INSTALL_NAME_DIR ${CMAKE_INSTALL_RPATH})

# hack to force cmake always create install and clean targets 
install(FILES DESTINATION)
# wrvb 2014-11-02: I don't think these are necessary to generate a clean 
#   target, the above line alone does it for me.  
# string(RANDOM LENGTH 32 __rand_target__name__)
# add_custom_target(${__rand_target__name__})
# unset(__rand_target__name__)

