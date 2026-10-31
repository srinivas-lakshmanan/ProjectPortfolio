# CMake generated Testfile for 
# Source directory: D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation
# Build directory: D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(arm_solver_tests "D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/build/Debug/arm_solver_tests.exe")
  set_tests_properties(arm_solver_tests PROPERTIES  _BACKTRACE_TRIPLES "D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/CMakeLists.txt;29;add_test;D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(arm_solver_tests "D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/build/Release/arm_solver_tests.exe")
  set_tests_properties(arm_solver_tests PROPERTIES  _BACKTRACE_TRIPLES "D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/CMakeLists.txt;29;add_test;D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(arm_solver_tests "D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/build/MinSizeRel/arm_solver_tests.exe")
  set_tests_properties(arm_solver_tests PROPERTIES  _BACKTRACE_TRIPLES "D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/CMakeLists.txt;29;add_test;D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(arm_solver_tests "D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/build/RelWithDebInfo/arm_solver_tests.exe")
  set_tests_properties(arm_solver_tests PROPERTIES  _BACKTRACE_TRIPLES "D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/CMakeLists.txt;29;add_test;D:/Srinivas/SrinivasLakshmanan_Resume/ProjectPortfolio/Differentiable_rigid_body_simulation/CMakeLists.txt;0;")
else()
  add_test(arm_solver_tests NOT_AVAILABLE)
endif()
