cmake_minimum_required(VERSION 3.17)

project(
		RayMLVQ
		VERSION 0.1
		DESCRIPTION "The MLVQ library with OptiX ray tracer"
		)

include(GenerateExportHeader)

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
file(GLOB RAYMLVQ_SOURCES_LOCAL "src/lib/*.cpp")

add_library(rayMLVQ
	SHARED
	${RAYMLVQ_SOURCES_LOCAL}
	)
set(RAYMLVQ_PCH_LOCAL 
	${UNIENGINE_PCH}
	${CMAKE_CURRENT_SOURCE_DIR}/include/lib/RayMLVQ-pch.hpp)
target_precompile_headers(rayMLVQ
	PRIVATE
	${RAYMLVQ_PCH_LOCAL}
	)
generate_export_header(rayMLVQ
	BASE_NAME RAYMLVQ
	EXPORT_MACRO_NAME RAYMLVQ_API
	)
# Linker settings for all platforms
target_link_libraries(rayMLVQ
	uniengine
	)

set(RAYMLVQ_INCLUDES_LOCAL
	${CMAKE_CURRENT_SOURCE_DIR}/include/lib
	${UNIENGINE_INCLUDES}
	${CMAKE_CURRENT_BINARY_DIR}
	)

target_include_directories(rayMLVQ
	PRIVATE
	${RAYMLVQ_INCLUDES_LOCAL}
	)

if (WIN32)
	# Compiler settings for Windows platform
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
	target_compile_definitions(rayMLVQ 
		PRIVATE 
		RAYMLVQ_EXPORTS
		)
elseif (UNIX AND NOT APPLE)
	if (${CMAKE_BUILD_TYPE} STREQUAL Release)
		set(CMAKE_CXX_FLAGS_RELEASE "-O3")
	elseif (${CMAKE_BUILD_TYPE} STREQUAL Debug)
		set(CMAKE_CXX_FLAGS_DEBUG "-g")
	endif ()
elseif (APPLE)
endif ()

# Compiler-specific compile flags
if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
	message(STATUS "Setting G++ flags")
	# G++
	target_compile_options(rayMLVQ PRIVATE -Wall -Wextra)
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
	message(STATUS "Setting MSVC flags")
	# MSVC
	target_compile_options(rayMLVQ PRIVATE /EHsc /W2 /c)
	
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
	message(STATUS "Setting Clang flags")
	# Activate all clang warnings except those that don't really affect much
	#target_compile_options(uniengine PRIVATE -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-newline-eof -Wno-padded -Wno-exit-time-destructors -Wno-global-constructors -Wno-constant-conversion)
endif ()

# ------------------------------------------------------------------
# Sorghum Factory - Provides models for ray tracer.
# ------------------------------------------------------------------
file(GLOB SORGHUM_FACTORY_SOURCES_LOCAL "src/core/*.cpp")

add_library(sorghumfactory
    ${SORGHUM_FACTORY_SOURCES_LOCAL}
)
target_precompile_headers(sorghumfactory
	PRIVATE
	${RAYMLVQ_PCH_LOCAL}
	)
target_include_directories(sorghumfactory
    PUBLIC
	${RAYMLVQ_INCLUDES_LOCAL}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
)
target_link_libraries(sorghumfactory
	rayMLVQ
	)

add_executable(sorghumMLVQ
	src/app/sample-main.cpp
	)

target_include_directories(sorghumMLVQ
	PRIVATE
	${RAYMLVQ_INCLUDES_LOCAL}
	)

target_precompile_headers(sorghumMLVQ
	PRIVATE
	${RAYMLVQ_PCH_LOCAL}
	)

target_link_libraries(sorghumMLVQ
	rayMLVQ
	sorghumfactory
	)
#file(COPY src/app/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#set(RAYMLVQ_INCLUDES ${RAYMLVQ_INCLUDES_LOCAL} PARENT_SCOPE)
#set(RAYMLVQ_PCH ${RAYMLVQ_PCH_LOCAL} PARENT_SCOPE)