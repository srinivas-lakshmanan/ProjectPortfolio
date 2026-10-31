cmake_minimum_required(VERSION 3.16)
project(articulated_arm_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(arm_solver
    src/arm_model.cpp
    src/kinematics_solver.cpp
    src/dynamics_solver.cpp
    src/rollout.cpp
    src/autodiff_rollout.cpp
)
target_include_directories(arm_solver PUBLIC include)

if(MSVC)
    target_compile_options(arm_solver PRIVATE /W4 /WX)
else()
    target_compile_options(arm_solver PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(arm_demo app/main.cpp)
target_link_libraries(arm_demo PRIVATE arm_solver)

add_executable(pendulum_demo app/pendulum.cpp)
target_link_libraries(pendulum_demo PRIVATE arm_solver)

include(CTest)
enable_testing()
add_executable(arm_solver_tests tests/arm_solver_tests.cpp)
target_link_libraries(arm_solver_tests PRIVATE arm_solver)
add_test(NAME arm_solver_tests COMMAND arm_solver_tests)

add_executable(eight_stage_tests tests/eight_stage_tests.cpp)
target_link_libraries(eight_stage_tests PRIVATE arm_solver)
add_test(NAME eight_stage_tests COMMAND eight_stage_tests)