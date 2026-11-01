cmake_minimum_required(VERSION 3.15)
project(neural_mpc_vehicle_model CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Header-only vehicle model library.
add_library(vehicle_model INTERFACE)
target_include_directories(vehicle_model INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_definitions(vehicle_model INTERFACE _USE_MATH_DEFINES)
endif()

# Demo simulation.
add_executable(vehicle_demo src/main.cpp)
target_link_libraries(vehicle_demo PRIVATE vehicle_model)

# Sanity tests.
enable_testing()
add_executable(test_double_track_model test/test_double_track_model.cpp)
target_link_libraries(test_double_track_model PRIVATE vehicle_model)
add_test(NAME double_track_model COMMAND test_double_track_model)

option(VEHICLE_MODEL_WITH_CASADI "Build the CasADi MPC function wrapper" OFF)
if(VEHICLE_MODEL_WITH_CASADI)
    find_package(casadi 3.7 CONFIG REQUIRED)

    add_library(vehicle_model_casadi INTERFACE)
    target_link_libraries(vehicle_model_casadi INTERFACE vehicle_model casadi::casadi)

    add_executable(test_casadi_vehicle_model test/test_casadi_vehicle_model.cpp)
    target_link_libraries(test_casadi_vehicle_model PRIVATE vehicle_model_casadi)
    add_test(NAME casadi_vehicle_model COMMAND test_casadi_vehicle_model)

    if(WIN32)
        get_target_property(CASADI_RUNTIME_LOCATION casadi::casadi IMPORTED_LOCATION_RELEASE)
        if(NOT CASADI_RUNTIME_LOCATION)
            get_target_property(CASADI_RUNTIME_LOCATION casadi::casadi IMPORTED_LOCATION)
        endif()
        get_filename_component(CASADI_RUNTIME_DIR "${CASADI_RUNTIME_LOCATION}" DIRECTORY)
        add_custom_command(TARGET test_casadi_vehicle_model POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:casadi::casadi>
                $<TARGET_FILE_DIR:test_casadi_vehicle_model>)

        # MinGW SDKs may bundle newer runtime DLLs than the local compiler.
        foreach(runtime libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll)
            if(EXISTS "${CASADI_RUNTIME_DIR}/${runtime}")
                add_custom_command(TARGET test_casadi_vehicle_model POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${CASADI_RUNTIME_DIR}/${runtime}"
                        $<TARGET_FILE_DIR:test_casadi_vehicle_model>)
            endif()
        endforeach()
    endif()
endif()
