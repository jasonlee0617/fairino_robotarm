cmake_minimum_required(VERSION 3.5)
project(hand_eye_calibration)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(ros2_aruco_interfaces REQUIRED)

ament_python_install_package(hand_eye_calibration)

install(DIRECTORY launch config rviz docs
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

install (PROGRAMS
  scripts/aruco_marker_pose_publisher.py
  scripts/calibration_aruco_publisher.py
  scripts/evaluate_calibration.py
  scripts/follow_aruco_marker.py
  scripts/handeye_publisher.py
  scripts/visualize_aruco_marker.py
  scripts/capture_d435_profile.py
  hand_eye_calibration/auto_calibration_collector.py
  hand_eye_calibration/manual_calibration_assistant.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # Tests import this package plus source-tree workspace dependencies.
  set(_test_pythonpath
    "${CMAKE_CURRENT_SOURCE_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/../../myrobot_common_ws/myrobot_common:${CMAKE_CURRENT_SOURCE_DIR}/../easy_handeye2/easy_handeye2")
  ament_add_pytest_test(test_config test/test_config.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
  ament_add_pytest_test(test_collector test/test_collector.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
  ament_add_pytest_test(test_solver test/test_solver.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
  ament_add_pytest_test(test_vision test/test_vision.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
  ament_add_pytest_test(test_manual_calibration_assistant test/test_manual_calibration_assistant.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
  ament_add_pytest_test(test_launch_topology test/test_launch_topology.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
  ament_add_pytest_test(test_capture_d435_profile test/test_capture_d435_profile.py
    APPEND_ENV "PYTHONPATH=${_test_pythonpath}")
endif()

ament_package()
