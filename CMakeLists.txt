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


include(${CMAKE_MODULE_PATH}/configure_build_type.cmake)
include(${CMAKE_MODULE_PATH}/configure_optix.cmake)
mark_as_advanced(CUDA_SDK_ROOT_DIR)

include_directories(${OptiX_INCLUDE})

# ------------------------------------------------------------------
# RayMLVQ
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/lib)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${UNIENGINE_INCLUDES})
cuda_compile_and_embed(ILLUMINATION_ESTIMATION_PTX ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/IlluminationEstimation.cu)
cuda_compile_and_embed(DEBUG_RENDERING_PTX ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/DebugRendering.cu)
file(GLOB RAYMLVQ_SOURCES_LOCAL "src/lib/*.cpp")
add_library(rayMLVQ
	SHARED
	${ILLUMINATION_ESTIMATION_PTX}
	${DEBUG_RENDERING_PTX}
	${RAYMLVQ_SOURCES_LOCAL}
	${CMAKE_MODULE_PATH}/configure_optix.cmake
	${CMAKE_MODULE_PATH}/FindOptiX.cmake
	)

generate_export_header(rayMLVQ
	BASE_NAME RAYMLVQ
	EXPORT_MACRO_NAME RAYMLVQ_API
	)

message(STATUS OptiX_INCLUDE:${OptiX_INCLUDE})
message(STATUS CUDA_LIBRARIES:${CUDA_LIBRARIES})
message(STATUS CUDA_CUDA_LIBRARY:${CUDA_CUDA_LIBRARY})
# Linker settings for all platforms
target_link_libraries(rayMLVQ
	uniengine
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
	)
message(STATUS UNIENGINE_INCLUDES:${UNIENGINE_INCLUDES})
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
	set (CUDA_PROPAGATE_HOST_FLAGS ON)
elseif (UNIX AND NOT APPLE)
	if (${CMAKE_BUILD_TYPE} STREQUAL Release)
		set(CMAKE_CXX_FLAGS_RELEASE "-O3")
	elseif (${CMAKE_BUILD_TYPE} STREQUAL Debug)
		set(CMAKE_CXX_FLAGS_DEBUG "-g")
	endif ()
elseif (APPLE)
endif ()

# ------------------------------------------------------------------
# Sorghum Factory - Provides models for ray tracer.
# ------------------------------------------------------------------
file(GLOB SORGHUM_FACTORY_SOURCES_LOCAL "src/core/*.cpp")
add_library(plantFactory
	${SORGHUM_FACTORY_SOURCES_LOCAL}
)

set(RAYMLVQ_PCH_LOCAL 
	${UNIENGINE_PCH}
	${CMAKE_CURRENT_SOURCE_DIR}/include/core/SorghumMLVQ-pch.hpp
	)

target_precompile_headers(plantFactory
	PRIVATE
	${RAYMLVQ_PCH_LOCAL}
	)
target_include_directories(plantFactory
	PUBLIC
	${RAYMLVQ_INCLUDES_LOCAL}
	${CMAKE_CURRENT_SOURCE_DIR}/include/core
)
target_link_libraries(plantFactory
	rayMLVQ
	)

set(SORGHUMMLVQ_RESOURCE_DIR_LOCAL ${CMAKE_CURRENT_SOURCE_DIR}/Resources/)
set(SORGHUMMLVQ_RESOURCE_DEF SORGHUMMLVQ_RESOURCE_FOLDER=\"${SORGHUMMLVQ_RESOURCE_DIR_LOCAL}\")


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
target_compile_definitions(sorghumMLVQ
	PRIVATE
	${SORGHUMMLVQ_RESOURCE_DEF}
	)
target_link_libraries(sorghumMLVQ
	rayMLVQ
	plantFactory
	)

file(COPY src/app/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if (WIN32)
	add_custom_command(
			TARGET sorghumMLVQ
			POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy
			"${CMAKE_CURRENT_BINARY_DIR}/3rdParty/UniEngine-dev/uniengine.dll"
			$<TARGET_FILE_DIR:sorghumMLVQ>
	)
elseif (UNIX AND NOT APPLE)
endif()
set(RAYMLVQ_INCLUDES ${RAYMLVQ_INCLUDES_LOCAL} PARENT_SCOPE)
set(RAYMLVQ_PCH ${RAYMLVQ_PCH_LOCAL} PARENT_SCOPE)