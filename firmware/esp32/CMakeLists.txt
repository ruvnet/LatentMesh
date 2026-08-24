cmake_minimum_required(VERSION 3.22.1)

# The portable implementation is owned by c/**.  c/CMakeLists.txt detects an
# ESP-IDF build and registers itself as component `c`.  An integrator may point
# at a packaged copy without changing this firmware.
set(LATENTMESH_C_COMPONENT "${CMAKE_CURRENT_LIST_DIR}/../../c" CACHE PATH
    "Path to the LatentMesh Air portable C ESP-IDF component")
if(NOT EXISTS "${LATENTMESH_C_COMPONENT}/CMakeLists.txt")
    message(FATAL_ERROR
        "Portable C component missing. Set LATENTMESH_C_COMPONENT to c/**")
endif()
list(APPEND EXTRA_COMPONENT_DIRS "${LATENTMESH_C_COMPONENT}")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(latentmesh_air_node)
