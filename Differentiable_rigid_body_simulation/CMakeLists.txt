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
target_include_directories(arm_solver PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(arm_solver PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(arm_solver PRIVATE /W4 /WX)
else()
    target_compile_options(arm_solver PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

option(ARM_BUILD_DEMOS "Build the command-line demonstrations" ON)
if(ARM_BUILD_DEMOS)
    add_executable(arm_demo app/main.cpp)
    target_link_libraries(arm_demo PRIVATE arm_solver)

    add_executable(pendulum_demo app/pendulum.cpp)
    target_link_libraries(pendulum_demo PRIVATE arm_solver)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(arm_solver_tests tests/arm_solver_tests.cpp)
    target_link_libraries(arm_solver_tests PRIVATE arm_solver)
    add_test(NAME arm_solver_tests COMMAND arm_solver_tests)

    add_executable(eight_stage_tests tests/eight_stage_tests.cpp)
    target_link_libraries(eight_stage_tests PRIVATE arm_solver)
    add_test(NAME eight_stage_tests COMMAND eight_stage_tests)
endif()

option(ARM_BUILD_GUI "Build the interactive ImGui application (downloads ImGui and GLFW)" ON)
if(ARM_BUILD_GUI)
    enable_language(C)
    include(FetchContent)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE)
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.91.8
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(glfw imgui)

    find_package(OpenGL REQUIRED)

    add_library(imgui_glfw STATIC
        "${imgui_SOURCE_DIR}/imgui.cpp"
        "${imgui_SOURCE_DIR}/imgui_draw.cpp"
        "${imgui_SOURCE_DIR}/imgui_tables.cpp"
        "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
        "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
    )
    target_include_directories(imgui_glfw PUBLIC
        "${imgui_SOURCE_DIR}"
        "${imgui_SOURCE_DIR}/backends")
    target_link_libraries(imgui_glfw PUBLIC glfw OpenGL::GL)

    add_executable(arm_gui app/gui_main.cpp)
    target_link_libraries(arm_gui PRIVATE arm_solver imgui_glfw)
endif()