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
if(NOT WIN32)
	set(OptiX_INSTALL_DIR "~/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64")
endif()
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

enable_language(CUDA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/lib/MLVQ)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/lib)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${UNIENGINE_INCLUDES})

file(GLOB RAY_TRACER_FACILITY_SOURCES_LOCAL 
	"src/lib/*.cpp"
	"src/lib/MLVQ/*.cpp"
	)
file(GLOB RAY_TRACER_FACILITY_CUDA_SOURCES_LOCAL 
	"src/lib/*.cu"
	)

cuda_compile_and_embed(DEFAULT_RENDERING_PTX ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/ptx/DefaultRendering.cu)
cuda_compile_and_embed(ILLUMINATION_ESTIMATION_PTX ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/ptx/IlluminationEstimation.cu)

add_library(rayTracerFacility
	SHARED
	#Default
	${DEFAULT_RENDERING_PTX}
	${ILLUMINATION_ESTIMATION_PTX}
	#RayMLVQ
	${MLVQ_RENDERING_PTX}

	#C++
	${RAY_TRACER_FACILITY_SOURCES_LOCAL}
	${RAY_TRACER_FACILITY_CUDA_SOURCES_LOCAL}
	${CMAKE_MODULE_PATH}/configure_optix.cmake
	${CMAKE_MODULE_PATH}/FindOptiX.cmake
	)

generate_export_header(rayTracerFacility
	BASE_NAME RAY_TRACER_FACILITY
	EXPORT_MACRO_NAME RAY_TRACER_FACILITY_API
	)

if(NOT WIN32)
	message(STATUS CUDALIB=${CUDA_TOOLKIT_ROOT_DIR})
	set(CUDA_CUDA_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
endif()
message(STATUS OptiX_INCLUDE:${OptiX_INCLUDE})
message(STATUS CUDA_LIBRARIES:${CUDA_LIBRARIES})
message(STATUS CUDA_CUDA_LIBRARY:${CUDA_CUDA_LIBRARY})
# Linker settings for all platforms
target_link_libraries(rayTracerFacility
	uniengine
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
	)
message(STATUS UNIENGINE_INCLUDES:${UNIENGINE_INCLUDES})
set(RAY_TRACER_FACILITY_INCLUDES_LOCAL
	${OptiX_INCLUDE}
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CMAKE_CURRENT_SOURCE_DIR}/include/lib
	${CMAKE_CURRENT_SOURCE_DIR}/include/lib/MLVQ
	${UNIENGINE_INCLUDES}
	${CMAKE_CURRENT_BINARY_DIR}
	)

target_include_directories(rayTracerFacility
	PRIVATE
	${RAY_TRACER_FACILITY_INCLUDES_LOCAL}
	)

if (WIN32)
	# Compiler settings for Windows platform
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
	target_compile_definitions(rayTracerFacility 
		PRIVATE 
		RAY_TRACER_FACILITY_EXPORTS
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
# Plant Factory - Provides models for ray tracer.
# ------------------------------------------------------------------
file(GLOB PLANT_FACTORY_SOURCES_LOCAL "src/core/*.cpp")
add_library(plantFactory
	${PLANT_FACTORY_SOURCES_LOCAL}
)
set(PLANT_FACTORY_PCH_LOCAL 
	${UNIENGINE_PCH}
	${CMAKE_CURRENT_SOURCE_DIR}/include/core/SorghumMLVQ-pch.hpp
	)


target_precompile_headers(plantFactory
	PRIVATE
	${PLANT_FACTORY_PCH_LOCAL}
	)
target_include_directories(plantFactory
	PUBLIC
	${RAY_TRACER_FACILITY_INCLUDES_LOCAL}
	${CMAKE_CURRENT_SOURCE_DIR}/include/core
)
target_link_libraries(plantFactory
	rayTracerFacility
	)

set(PLANT_FACTORY_RESOURCE_DIR_LOCAL ${CMAKE_CURRENT_SOURCE_DIR}/Resources/)
set(PLANT_FACTORY_RESOURCE_DEF PLANT_FACTORY_RESOURCE_FOLDER=\"${PLANT_FACTORY_RESOURCE_DIR_LOCAL}\")


add_executable(sorghumMLVQ
	"src/app/SorghumMLVQ.cpp"
	)

target_include_directories(sorghumMLVQ
	PRIVATE
	${RAY_TRACER_FACILITY_INCLUDES_LOCAL}
	)

target_precompile_headers(sorghumMLVQ
	PRIVATE
	${PLANT_FACTORY_PCH_LOCAL}
	)
target_compile_definitions(sorghumMLVQ
	PRIVATE
	${PLANT_FACTORY_RESOURCE_DEF}
	)
target_link_libraries(sorghumMLVQ
	rayTracerFacility
	plantFactory
	)

file(COPY src/app/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(NOT ${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_BINARY_DIR})
	if (WIN32)
		add_custom_command(
				TARGET rayTracerFacility
				POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy
				${PROJECT_BINARY_DIR}/rayTracerFacility.dll
				${CMAKE_BINARY_DIR}
		)
	endif()
	set(RAY_TRACER_FACILITY_INCLUDES ${RAY_TRACER_FACILITY_INCLUDES_LOCAL} PARENT_SCOPE)
	set(RAY_TRACER_FACILITY_PCH ${PLANT_FACTORY_PCH_LOCAL} PARENT_SCOPE)
endif()
