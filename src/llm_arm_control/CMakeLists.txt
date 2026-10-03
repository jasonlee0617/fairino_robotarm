cmake_minimum_required(VERSION 3.8)
project(llm_arm_control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/AgentCommand.srv"
  "action/ExecutePreview.action"
  DEPENDENCIES geometry_msgs
)

ament_python_install_package(llm_arm_control_nodes)

install(PROGRAMS llm_arm_control_nodes/llm_control_task_server.py
  DESTINATION lib/${PROJECT_NAME} RENAME llm_control_task_server)
install(PROGRAMS llm_arm_control_nodes/voice_wake_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME voice_wake_node)
install(PROGRAMS llm_arm_control_nodes/voice_realtime_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME voice_realtime_node)
install(PROGRAMS scripts/setup_voice_runtime.sh
  DESTINATION lib/${PROJECT_NAME})
install(
  DIRECTORY audio config model rviz launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(ENV{PYTEST_DISABLE_PLUGIN_AUTOLOAD} "1")
  set(ENV{PYTHONNOUSERSITE} "1")
  foreach(test_name
      test_voice_control
      test_agent_command
      test_realtime_architecture
      test_llm_robot_control_sim_launch
      test_llm_control_state_machine
      test_shutdown
      test_task_logic)
    ament_add_pytest_test(
      ${test_name}
      test/${test_name}.py
      PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
      ENV
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
        "PYTHONNOUSERSITE=1"
    )
  endforeach()
endif()

ament_package()
