cmake_minimum_required(VERSION 3.8)
project(myrobot_simulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(test_d435_profile test/test_d435_profile.py)
  ament_add_pytest_test(test_motion_planning_sim test/test_motion_planning_sim.py)
  ament_add_pytest_test(test_pathplanning_scene_tools test/test_pathplanning_scene_tools.py)
  ament_add_pytest_test(test_adaptive_goal_geometry test/test_adaptive_goal_geometry.py)
  ament_add_pytest_test(test_planning_benchmark_artifacts test/test_planning_benchmark_artifacts.py)
  ament_add_pytest_test(test_mpc_avoidance_node_sim test/test_mpc_avoidance_node_sim.py)
  ament_add_pytest_test(test_visual_grasping_myrobot_simulation test/test_visual_grasping_myrobot_simulation.py)
endif()

install(
  DIRECTORY
    scripts
    launch
    launch_utils
    config
    rviz
    worlds
    docs
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
  scripts/pick_drop_node_sim.py
  scripts/target_motion_controller_node.py
  scripts/motion_planning_node_sim.py
  scripts/mpc_avoidance_node_sim.py

  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES
  scripts/pathplanning_scene_tools.py
  scripts/planning_benchmark.py
  scripts/planning_motion.py
  scripts/planning_trace.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
