###############################################################################
# The AEMTemplate Application
################################################################################

# Define CMake compatibility version
cmake_minimum_required (VERSION 3.0 FATAL_ERROR)

# The name of this project.
project (BackpropPorting)

list (APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set (CMAKE_INSTALL_RPATH "${BOSP_SYSROOT}/lib/bbque;${BOSP_SYSROOT}/lib")
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Make sure the compiler can find include files
include_directories (
	${PROJECT_BINARY_DIR}/include
	${PROJECT_SOURCE_DIR}/include
)

# Add compilation dependencies
include_directories(
	${BOSP_SYSROOT}/include
)

# Add linking dependencies
link_directories(
	${BOSP_SYSROOT}/lib
	${BOSP_SYSROOT}/lib/bbque
)

# Recurse into project subfolders
add_subdirectory (src)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/recipes/"
	DESTINATION "${BOSP_SYSROOT}/etc/bbque/recipes"
	FILES_MATCHING PATTERN "*.recipe")

################################################################################
# Configuration resume
################################################################################

message ( STATUS )
message ( STATUS "===================[ AEMTemplate - Building System Configuration ]"
                 "====================" )
message ( STATUS "Build Flavor.......... ${CMAKE_BUILD_TYPE}" )
message ( STATUS "   CFLAGS............. ${CMAKE_C_FLAGS}")
message ( STATUS " CXXFLAGS............. ${CMAKE_CXX_FLAGS}")
message ( STATUS "Build directory....... ${CMAKE_BINARY_DIR}")
message ( STATUS "Installation prefix... ${CMAKE_INSTALL_PREFIX}" )
message ( STATUS "Using RTLib........... ${BBQUE_RTLIB_LIBRARY}" )
message ( STATUS )
message ( STATUS "Default values could be changes at command line, with:")
message ( STATUS "  cmake -D<Variable>=<Value>" )
message ( STATUS "or use the '-i' cmake flag to run in interactive mode" )
message ( STATUS "======================================================"
                 "==========================" )
message(STATUS)

