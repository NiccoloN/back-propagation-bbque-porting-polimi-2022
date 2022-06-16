
cmake_minimum_required(VERSION 3.0)

find_library(BBQUE_RTLIB_LIBRARY bbque_rtlib
	HINTS ${BOSP_SYSROOT}
	PATHS ${BOSP_SYSROOT}
	PATH_SUFFIXES lib/bbque
)

# Check for required libs and packages (headers + lib)
set(Boost_DIR /home/nico/Desktop/bosp/out/usr/lib/cmake/Boost-1.78.0)
find_package(Boost 1.78.0 REQUIRED program_options
	PATHS ${BOSP_SYSROOT})
message (STATUS "Boost libraries: ${Boost_LIBRARIES}")
message (STATUS "Boost include  : ${Boost_INCLUDE_DIRS}")

find_package(OpenMP REQUIRED)

find_library(DLB libdlb.a)

set(BACKPROP_PORTING_SRC BackpropPorting_exc BackpropPorting_main backprop.c)
add_executable(backprop-porting ${BACKPROP_PORTING_SRC})

set_property(TARGET backprop-porting PROPERTY CXX_STANDARD 11)

target_link_libraries(
		backprop-porting
		${BBQUE_RTLIB_LIBRARY}
		${Boost_LIBRARIES}
		${OpenMP_CXX_FLAGS}
		${DLB}
)

target_compile_options(backprop-porting PRIVATE ${OpenMP_CXX_FLAGS})

# Use link path ad RPATH
set_property(TARGET backprop-porting PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)

install (TARGETS backprop-porting RUNTIME
	DESTINATION ${CMAKE_INSTALL_PREFIX}/usr/bin)

