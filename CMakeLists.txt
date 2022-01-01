cmake_minimum_required(VERSION 3.17)
project(
        SorghumFactory
        VERSION 0.1
        DESCRIPTION "An implementation of MLVQ library in CUDA with OptiX ray tracer, and a procedural 3D model of sorghum. The project is based on UniEngine."
)

option(BUILD_RAY_TRACER_FACILITY "Build Ray Tracer Facility" ON)
set(BUILD_RAY_TRACER_FACILITY ON)
include(GenerateExportHeader)

# Set a default build type if none was specified
set(default_build_type "Release")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release")
endif ()

# ------------------------------------------------------------------
# 3rd Party libraries
# ------------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdParty)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 17)
if (WIN32)
    # Compiler settings for Windows platform
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
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
file(GLOB SORGHUM_FACTORY_SOURCES_LOCAL "src/SorghumFactory/*.cpp")
add_library(SorghumFactory
        SHARED
        ${SORGHUM_FACTORY_SOURCES_LOCAL}
        )
set(SORGHUM_FACTORY_PCH_LOCAL
        ${UNIENGINE_PCH}
        ${RAY_TRACER_FACILITY_PCH}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/SorghumFactory/SorghumFactory-pch.hpp
        )
generate_export_header(SorghumFactory
        BASE_NAME SORGHUM_FACTORY
        EXPORT_MACRO_NAME SORGHUM_FACTORY_API
        )

target_precompile_headers(SorghumFactory
        PRIVATE
        ${SORGHUM_FACTORY_PCH_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_compile_definitions(SorghumFactory
            PRIVATE
            GLAD_GLAPI_EXPORT
            NOMINMAX
            RAYTRACERFACILITY
            )
    set(SORGHUM_FACTORY_INCLUDES_LOCAL
            ${RAY_TRACER_FACILITY_INCLUDES}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/SorghumFactory
            ${CMAKE_CURRENT_SOURCE_DIR}/include/SorghumFactory/quickhull
            ${CMAKE_CURRENT_SOURCE_DIR}/include/SorghumFactory/rapidxml
            )
else ()
    target_compile_definitions(SorghumFactory
            PRIVATE
            GLAD_GLAPI_EXPORT
            NOMINMAX
            )
    set(SORGHUM_FACTORY_INCLUDES_LOCAL
            ${UNIENGINE_INCLUDES}
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/SorghumFactory
            ${CMAKE_CURRENT_SOURCE_DIR}/include/SorghumFactory/quickhull
            ${CMAKE_CURRENT_SOURCE_DIR}/include/SorghumFactory/rapidxml
            )
endif ()
target_include_directories(SorghumFactory
        PUBLIC
        ${SORGHUM_FACTORY_INCLUDES_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_link_libraries(SorghumFactory
            RayTracerFacility
            )
else ()
    target_link_libraries(SorghumFactory
            uniengine
            )
endif ()
# ------------------------------------------------------------------
# Scripts
# ------------------------------------------------------------------
file(GLOB SCRIPTS_SOURCES_LOCAL "src/Scripts/*.cpp")
add_library(Scripts
        ${SCRIPTS_SOURCES_LOCAL}
        )
target_precompile_headers(Scripts
        PRIVATE
        ${SORGHUM_FACTORY_PCH_LOCAL}
        )
set(SCRIPTS_INCLUDES_LOCAL
        ${SORGHUM_FACTORY_INCLUDES_LOCAL}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/Scripts
        )
target_include_directories(Scripts
        PUBLIC
        ${SCRIPTS_INCLUDES_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_link_libraries(Scripts
            RayTracerFacility
            SorghumFactory
            )
    target_compile_definitions(Scripts
            PRIVATE
            RAYTRACERFACILITY
            )
else ()
    target_link_libraries(Scripts
            uniengine
            SorghumFactory
            )
endif ()
# ------------------------------------------------------------------
# Executables
# ------------------------------------------------------------------

add_executable(Application
        "src/app/Application.cpp"
        )
target_include_directories(Application
        PUBLIC
        ${SCRIPTS_INCLUDES_LOCAL}
        )
target_precompile_headers(Application
        PRIVATE
        ${SORGHUM_FACTORY_PCH_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_link_libraries(Application
            RayTracerFacility
            SorghumFactory
            Scripts
            )
    target_compile_definitions(Application
            PRIVATE
            RAYTRACERFACILITY
            )
else ()
    target_link_libraries(Application
            uniengine
            SorghumFactory
            Scripts
            )
endif ()
add_executable(DatasetPipelines
        "src/app/DatasetPipelines.cpp"
        )
target_include_directories(DatasetPipelines
        PUBLIC
        ${SCRIPTS_INCLUDES_LOCAL}
        )
target_precompile_headers(DatasetPipelines
        PRIVATE
        ${SORGHUM_FACTORY_PCH_LOCAL}
        )
if (BUILD_RAY_TRACER_FACILITY)
    target_link_libraries(DatasetPipelines
            RayTracerFacility
            SorghumFactory
            Scripts
            )
    target_compile_definitions(DatasetPipelines
            PRIVATE
            RAYTRACERFACILITY
            )
else ()
    target_link_libraries(DatasetPipelines
            uniengine
            SorghumFactory
            Scripts
            )
endif ()

# ------------------------------------------------------------------
# Copy Internal resources
# ------------------------------------------------------------------
add_custom_command(TARGET SorghumFactory POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Internals
        ${CMAKE_BINARY_DIR})

file(COPY src/app/imgui.ini DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
