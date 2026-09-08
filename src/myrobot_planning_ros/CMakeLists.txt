cmake_minimum_required(VERSION 3.16)
project(myrobot_planning_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ─── 依赖 ───
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(myrobot_planning_core REQUIRED)

set(THIS_PACKAGE_DEPS
  rclcpp
  rclcpp_action
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  pluginlib
  tf2_ros
  tf2_eigen
  trajectory_msgs
  moveit_msgs
  sensor_msgs
  geometry_msgs
  std_msgs
  myrobot_planning_core
)

# ─── 源文件 ───
set(ALL_ROS_SOURCES
  src/config/parameter_loader.cpp
  src/moveit_collision_checker.cpp
  src/fairino_planner_manager.cpp
  src/fairino_ik_plugin.cpp
  src/pipeline/fairino_planning_pipeline.cpp
  
)

set(ROS_SOURCES "")
foreach(src_file ${ALL_ROS_SOURCES})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
    list(APPEND ROS_SOURCES ${src_file})
  else()
    message(WARNING "Source file not found, skipping: ${src_file}")
  endif()
endforeach()

if(NOT ROS_SOURCES)
  message(FATAL_ERROR "No source files found in src/!")
endif()

# ─── 插件库 ───
add_library(${PROJECT_NAME} SHARED ${ROS_SOURCES})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

# ★ 去掉 PUBLIC 关键字
ament_target_dependencies(${PROJECT_NAME} ${THIS_PACKAGE_DEPS})

# ─── 独立规划节点 ───
set(STANDALONE_SRC src/standalone_planner_node.cpp)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${STANDALONE_SRC}")
  add_executable(standalone_planner ${STANDALONE_SRC})

  target_include_directories(standalone_planner PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  )

  ament_target_dependencies(standalone_planner ${THIS_PACKAGE_DEPS})
  target_link_libraries(standalone_planner ${PROJECT_NAME})

  install(TARGETS standalone_planner
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
else()
  message(WARNING "standalone_planner_node.cpp not found, skipping executable")
endif()

set(FAIRINO_CARTESIAN_SRC src/fairino_cartesian_path_node.cpp)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${FAIRINO_CARTESIAN_SRC}")
  add_executable(fairino_cartesian_path_server ${FAIRINO_CARTESIAN_SRC})

  target_include_directories(fairino_cartesian_path_server PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  )

  ament_target_dependencies(fairino_cartesian_path_server ${THIS_PACKAGE_DEPS})
  target_link_libraries(fairino_cartesian_path_server ${PROJECT_NAME})

  install(TARGETS fairino_cartesian_path_server
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

# ─── MoveIt2 插件声明 ───

set(PLUGIN_XML plugins/myrobot_planning_plugins.xml)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${PLUGIN_XML}")
  pluginlib_export_plugin_description_file(moveit_core ${PLUGIN_XML})
  install(FILES ${PLUGIN_XML} DESTINATION share/${PROJECT_NAME})
else()
  message(WARNING "Plugin XML not found: ${PLUGIN_XML}")
endif()

# ─── 安装 ───
install(DIRECTORY include/ DESTINATION include)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  install(DIRECTORY docs/ DESTINATION share/${PROJECT_NAME}/docs)
endif()

# ─── 测试 ───
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_moveit_collision_checker
    test/test_moveit_collision_checker.cpp
  )
  target_link_libraries(test_moveit_collision_checker ${PROJECT_NAME})
  ament_target_dependencies(test_moveit_collision_checker ${THIS_PACKAGE_DEPS})
endif()

# ─── ament 导出 ───
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_DEPS})
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})

ament_package()
