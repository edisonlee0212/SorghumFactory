cmake_minimum_required(VERSION 3.17)

project(
		RayMLVQ
		VERSION 0.1
		DESCRIPTION "The MLVQ library with OptiX ray tracer"
		)

# ------------------------------------------------------------------
# 3rd Party libraries
# ------------------------------------------------------------------
# We need a CMAKE_DIR with some code to find external dependencies
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(3RD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${3RD_PARTY_DIR})

# ------------------------------------------------------------------
# RayMLVQ
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)

set(RAYMLVQ_PCH_LOCAL ${CMAKE_CURRENT_SOURCE_DIR}/include/lib/RayMLVQ-pch.hpp)

file(GLOB RAYMLVQ_SOURCES_LOCAL "src/lib/*.cpp")

add_library(rayMLVQ-lib 
	SHARED
	${RAYMLVQ_SOURCES_LOCAL}
	)
generate_export_header(rayMLVQ-lib
	BASE_NAME rayMLVQlib
	EXPORT_MACRO_NAME RAYMLVQ_API
	)
# Linker settings for all platforms
target_link_libraries(rayMLVQ-lib
	uniengine-lib
	)
