cmake_minimum_required(VERSION 3.16)
project(myrobot_planning_core)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

# ─── 收集实际存在的源文件 ───
# 先列出所有预期源文件, 再过滤掉不存在的
set(ALL_EXPECTED_SOURCES
  src/dh_kinematics.cpp
  src/ik/fairino_ik.cpp
  src/ik/ik_selector.cpp
  src/ik/cartesian_path_planner.cpp
  src/tree/rrt_tree.cpp
  src/aapf/aapf_guided_sampler.cpp
  src/constraints/orientation_checker.cpp
  src/trajectory/path_shortcut.cpp
  src/trajectory/trajectory_smoother.cpp
  src/algorithms/bidirectional/aapf_bi_rrt_star.cpp
  src/algorithms/bidirectional/bi_rrt_star.cpp
  src/algorithms/graph/mire_bi_ait_star.cpp
  src/algorithms/graph/prm.cpp
  src/algorithms/rrt/rrt.cpp
  src/algorithms/rrt/rrt_star.cpp
  src/algorithms/rrt/rrt_star_search.cpp
  src/algorithms/rrt/informed_rrt_star.cpp
)

set(CORE_SOURCES "")
foreach(src_file ${ALL_EXPECTED_SOURCES})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
    list(APPEND CORE_SOURCES ${src_file})
  else()
    message(WARNING "Source file not found, skipping: ${src_file}")
  endif()
endforeach()

if(NOT CORE_SOURCES)
  message(FATAL_ERROR "No source files found in src/!")
endif()

message(STATUS "Building with sources: ${CORE_SOURCES}")

# ─── 库 ───
add_library(${PROJECT_NAME} SHARED ${CORE_SOURCES})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC Eigen3::Eigen
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# ─── 安装头文件 ───
install(DIRECTORY include/
  DESTINATION include
)

# ─── 安装库 ───
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  install(DIRECTORY docs/ DESTINATION share/${PROJECT_NAME}/docs)
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_aapf_bi_rrt_star test/test_aapf_bi_rrt_star.cpp)
  target_link_libraries(test_aapf_bi_rrt_star ${PROJECT_NAME})
  ament_add_gtest(test_aapf_guided_sampler test/test_aapf_guided_sampler.cpp)
  target_link_libraries(test_aapf_guided_sampler ${PROJECT_NAME})
  ament_add_gtest(test_bi_rrt_star test/test_bi_rrt_star.cpp)
  target_link_libraries(test_bi_rrt_star ${PROJECT_NAME})
  ament_add_gtest(test_mire_bi_ait_star test/test_mire_bi_ait_star.cpp)
  target_link_libraries(test_mire_bi_ait_star ${PROJECT_NAME})
  ament_add_gtest(test_rrt_tree test/test_rrt_tree.cpp)
  target_link_libraries(test_rrt_tree ${PROJECT_NAME})
  ament_add_gtest(test_rrt test/test_rrt.cpp)
  target_link_libraries(test_rrt ${PROJECT_NAME})
  ament_add_gtest(test_informed_rrt_star test/test_informed_rrt_star.cpp)
  target_link_libraries(test_informed_rrt_star ${PROJECT_NAME})
  ament_add_gtest(test_prm test/test_prm.cpp)
  target_link_libraries(test_prm ${PROJECT_NAME})
  ament_add_gtest(test_ik_selector_profiles test/test_ik_selector_profiles.cpp)
  target_link_libraries(test_ik_selector_profiles ${PROJECT_NAME})
  ament_add_gtest(test_dh_tcp_transform test/test_dh_tcp_transform.cpp)
  target_link_libraries(test_dh_tcp_transform ${PROJECT_NAME})
endif()

# ─── ament 导出 ───
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})

ament_package()
