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

    add_executable(mpc_maneuvers src/mpc_maneuvers.cpp)
    target_link_libraries(mpc_maneuvers PRIVATE vehicle_model_casadi)
    add_executable(test_maneuver_reference test/test_maneuver_reference.cpp)
    target_link_libraries(test_maneuver_reference PRIVATE vehicle_model_casadi)
    add_test(NAME maneuver_reference COMMAND test_maneuver_reference)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mpc_results")
    add_test(NAME mpc_step_steer COMMAND mpc_maneuvers
        --scenario step-steer --output-dir "${CMAKE_CURRENT_BINARY_DIR}/mpc_results")
    add_test(NAME mpc_double_lane_change COMMAND mpc_maneuvers
        --scenario double-lane-change --output-dir "${CMAKE_CURRENT_BINARY_DIR}/mpc_results")
    set_tests_properties(mpc_step_steer mpc_double_lane_change PROPERTIES
        LABELS "mpc;closed-loop" TIMEOUT 900
        ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")

    if(WIN32)
        get_target_property(CASADI_RUNTIME_LOCATION casadi::casadi IMPORTED_LOCATION_RELEASE)
        if(NOT CASADI_RUNTIME_LOCATION)
            get_target_property(CASADI_RUNTIME_LOCATION casadi::casadi IMPORTED_LOCATION)
        endif()
        get_filename_component(CASADI_RUNTIME_DIR "${CASADI_RUNTIME_LOCATION}" DIRECTORY)
        foreach(target test_casadi_vehicle_model test_maneuver_reference mpc_maneuvers)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:casadi::casadi>
                    $<TARGET_FILE_DIR:${target}>)

            # Core runtime and IPOPT dependencies supplied by MinGW CasADi SDKs.
            foreach(runtime libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll
                    libcasadi_nlpsol_ipopt.dll libipopt-3.dll libcoinmumps-3.dll
                    libcoinmetis-2.dll libcasadi-tp-openblas.dll libgfortran-5.dll
                    libquadmath-0.dll libcasadi_importer_shell.dll)
                if(EXISTS "${CASADI_RUNTIME_DIR}/${runtime}")
                    add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${CASADI_RUNTIME_DIR}/${runtime}"
                        $<TARGET_FILE_DIR:${target}>)
                endif()
            endforeach()
        endforeach()
    endif()
endif()
