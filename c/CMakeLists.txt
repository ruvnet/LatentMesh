cmake_minimum_required(VERSION 3.16)

set(LM_AIR_SOURCES
    src/air.c
    src/fec.c
    src/frame.c
    src/modem.c
    src/neural.c)

if(COMMAND idf_component_register)
    idf_component_register(
        SRCS ${LM_AIR_SOURCES}
        INCLUDE_DIRS include)
    target_compile_features(${COMPONENT_LIB} PUBLIC c_std_11)
    target_link_libraries(${COMPONENT_LIB} PUBLIC m)
    return()
endif()

project(latentmesh_air VERSION 0.1.0 LANGUAGES C)

option(LM_AIR_BUILD_TESTS "Build unit and malformed-input tests" ON)
option(LM_AIR_BUILD_EXAMPLES "Build command-line loopback" ON)
option(LM_AIR_BUILD_BENCHMARKS "Build microbenchmarks" ON)
option(LM_AIR_ENABLE_SANITIZERS "Enable address and undefined sanitizers" OFF)

add_library(latentmesh_air STATIC ${LM_AIR_SOURCES})
add_library(latentmesh::air ALIAS latentmesh_air)
target_include_directories(latentmesh_air
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_compile_features(latentmesh_air PUBLIC c_std_11)
target_link_libraries(latentmesh_air PUBLIC m)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(latentmesh_air PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(LM_AIR_ENABLE_SANITIZERS)
        target_compile_options(latentmesh_air PUBLIC
            -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(latentmesh_air PUBLIC
            -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
endif()

if(LM_AIR_BUILD_TESTS)
    enable_testing()
    add_executable(latentmesh_air_tests tests/test_air.c)
    target_link_libraries(latentmesh_air_tests PRIVATE latentmesh_air)
    add_test(NAME latentmesh_air_tests COMMAND latentmesh_air_tests)
endif()

if(LM_AIR_BUILD_EXAMPLES)
    add_executable(latentmesh-air-loopback examples/loopback.c)
    target_link_libraries(latentmesh-air-loopback PRIVATE latentmesh_air)
endif()

if(LM_AIR_BUILD_BENCHMARKS)
    add_executable(latentmesh-air-bench bench/bench.c)
    target_link_libraries(latentmesh-air-bench PRIVATE latentmesh_air)
endif()

include(GNUInstallDirs)
install(TARGETS latentmesh_air EXPORT LatentMeshAirTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT LatentMeshAirTargets
    NAMESPACE latentmesh::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LatentMeshAir)
