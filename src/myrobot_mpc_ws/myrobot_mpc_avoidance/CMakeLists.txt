cmake_minimum_required(VERSION 3.8)
project(myrobot_mpc_avoidance)

set(CMAKE_CXX_STANDARD 17)
option(BUILD_MPC_SIM_TOOLS "Build simulation/demo runtime tools" ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(myrobot_planning_core REQUIRED)

# ignition-transport (Gazebo 直接通信，使用 pkg-config)
find_package(PkgConfig REQUIRED)
pkg_check_modules(IGNITION_TRANSPORT REQUIRED ignition-transport11)

# acados: this workspace owns its pinned toolbox.
get_filename_component(MPC_TOOLBOX_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../mpc_toolbox" ABSOLUTE)
set(ACADOS_DIR "${MPC_TOOLBOX_DIR}/acados")
file(TO_CMAKE_PATH "${ACADOS_DIR}" ACADOS_DIR)

set(ACADOS_CONFIGURED_DIR_PREV "")
if(DEFINED FAIRINO_ACADOS_CONFIGURED_DIR)
    set(ACADOS_CONFIGURED_DIR_PREV "${FAIRINO_ACADOS_CONFIGURED_DIR}")
endif()
if(ACADOS_CONFIGURED_DIR_PREV AND NOT ACADOS_CONFIGURED_DIR_PREV STREQUAL ACADOS_DIR)
    message(FATAL_ERROR
        "ACADOS_INSTALL_DIR changed for an existing build tree.\n"
        "  cached: ${ACADOS_CONFIGURED_DIR_PREV}\n"
        "  current: ${ACADOS_DIR}\n"
        "Please remove build/myrobot_mpc_avoidance and install/myrobot_mpc_avoidance, "
        "then rerun colcon build after sourcing scripts/setup_acados_env.sh."
    )
endif()
set(FAIRINO_ACADOS_CONFIGURED_DIR "${ACADOS_DIR}" CACHE INTERNAL
    "ACADOS path used to configure myrobot_mpc_avoidance")
message(STATUS "myrobot_mpc_avoidance using ACADOS_DIR=${ACADOS_DIR}")

set(ACADOS_GEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code")
set(ACADOS_GEN_MODEL_DIR "${ACADOS_GEN_ROOT}/fairino_arm_mpc_model")
set(ACADOS_GEN_COST_DIR "${ACADOS_GEN_ROOT}/fairino_arm_mpc_cost")
set(ACADOS_GEN_CONSTR_DIR "${ACADOS_GEN_ROOT}/fairino_arm_mpc_constraints")
set(ACADOS_NMPC_GEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code_nmpc")
set(ACADOS_NMPC_GEN_MODEL_DIR "${ACADOS_NMPC_GEN_ROOT}/fairino_arm_nmpc_model")
set(ACADOS_NMPC_GEN_COST_DIR "${ACADOS_NMPC_GEN_ROOT}/fairino_arm_nmpc_cost")
set(ACADOS_NMPC_GEN_CONSTR_DIR "${ACADOS_NMPC_GEN_ROOT}/fairino_arm_nmpc_constraints")

set(ACADOS_GENERATED_CORE
    "${ACADOS_GEN_ROOT}/acados_solver_fairino_arm_mpc.c"
    "${ACADOS_GEN_ROOT}/acados_solver_fairino_arm_mpc.h"
    "${ACADOS_GEN_ROOT}/acados_sim_solver_fairino_arm_mpc.c"
    "${ACADOS_GEN_ROOT}/acados_sim_solver_fairino_arm_mpc.h"
    "${ACADOS_GEN_ROOT}/acados_solver.pxd"
    "${ACADOS_GEN_ROOT}/main_fairino_arm_mpc.c"
    "${ACADOS_GEN_ROOT}/main_sim_fairino_arm_mpc.c"
    "${ACADOS_GEN_ROOT}/Makefile"
)

set(ACADOS_GENERATED_MODEL
    "${ACADOS_GEN_MODEL_DIR}/fairino_arm_mpc_expl_ode_fun.c"
    "${ACADOS_GEN_MODEL_DIR}/fairino_arm_mpc_expl_vde_adj.c"
    "${ACADOS_GEN_MODEL_DIR}/fairino_arm_mpc_expl_vde_forw.c"
    "${ACADOS_GEN_MODEL_DIR}/fairino_arm_mpc_model.h"
)

set(ACADOS_GENERATED_COST
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_0_fun.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_0_fun_jac.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_0_fun_jac_hess.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_fun.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_fun_jac.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_fun_jac_hess.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_e_fun.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_e_fun_jac.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_e_fun_jac_hess.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost.h"
)

set(ACADOS_GENERATED_CONSTR
    "${ACADOS_GEN_CONSTR_DIR}/fairino_arm_mpc_constr_h_fun.c"
    "${ACADOS_GEN_CONSTR_DIR}/fairino_arm_mpc_constr_h_fun_jac_uxt_zt.c"
    "${ACADOS_GEN_CONSTR_DIR}/fairino_arm_mpc_constraints.h"
)

set(ACADOS_GENERATED_FILES
    ${ACADOS_GENERATED_CORE}
    ${ACADOS_GENERATED_MODEL}
    ${ACADOS_GENERATED_COST}
    ${ACADOS_GENERATED_CONSTR}
)

set(ACADOS_NMPC_GENERATED_CORE
    "${ACADOS_NMPC_GEN_ROOT}/acados_solver_fairino_arm_nmpc.c"
    "${ACADOS_NMPC_GEN_ROOT}/acados_solver_fairino_arm_nmpc.h"
    "${ACADOS_NMPC_GEN_ROOT}/acados_sim_solver_fairino_arm_nmpc.c"
    "${ACADOS_NMPC_GEN_ROOT}/acados_sim_solver_fairino_arm_nmpc.h"
    "${ACADOS_NMPC_GEN_ROOT}/acados_solver.pxd"
    "${ACADOS_NMPC_GEN_ROOT}/main_fairino_arm_nmpc.c"
    "${ACADOS_NMPC_GEN_ROOT}/main_sim_fairino_arm_nmpc.c"
    "${ACADOS_NMPC_GEN_ROOT}/Makefile"
)

set(ACADOS_NMPC_GENERATED_MODEL
    "${ACADOS_NMPC_GEN_MODEL_DIR}/fairino_arm_nmpc_expl_ode_fun.c"
    "${ACADOS_NMPC_GEN_MODEL_DIR}/fairino_arm_nmpc_expl_vde_adj.c"
    "${ACADOS_NMPC_GEN_MODEL_DIR}/fairino_arm_nmpc_expl_vde_forw.c"
    "${ACADOS_NMPC_GEN_MODEL_DIR}/fairino_arm_nmpc_model.h"
)

set(ACADOS_NMPC_GENERATED_COST
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_0_fun.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_0_fun_jac.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_0_fun_jac_hess.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_fun.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_fun_jac.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_fun_jac_hess.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_e_fun.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_e_fun_jac.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_e_fun_jac_hess.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost.h"
)

set(ACADOS_NMPC_GENERATED_CONSTR
    "${ACADOS_NMPC_GEN_CONSTR_DIR}/fairino_arm_nmpc_constr_h_fun.c"
    "${ACADOS_NMPC_GEN_CONSTR_DIR}/fairino_arm_nmpc_constr_h_fun_jac_uxt_zt.c"
    "${ACADOS_NMPC_GEN_CONSTR_DIR}/fairino_arm_nmpc_constraints.h"
)

set(ACADOS_NMPC_GENERATED_FILES
    ${ACADOS_NMPC_GENERATED_CORE}
    ${ACADOS_NMPC_GENERATED_MODEL}
    ${ACADOS_NMPC_GENERATED_COST}
    ${ACADOS_NMPC_GENERATED_CONSTR}
)

add_custom_command(
    OUTPUT ${ACADOS_GENERATED_FILES}
    COMMAND bash -lc "source scripts/setup_acados_env.sh && python3 scripts/generate_acados_solver.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_acados_solver.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/acados_codegen_common.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_acados_env.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/acados_ocp_fairino_mpc.json"
    COMMENT "Generating acados solver sources into c_generated_code/"
    VERBATIM
)

add_custom_target(generate_acados_solver_code DEPENDS ${ACADOS_GENERATED_FILES})

add_custom_command(
    OUTPUT ${ACADOS_NMPC_GENERATED_FILES}
    COMMAND bash -lc "source scripts/setup_acados_env.sh && python3 scripts/generate_acados_nmpc_solver.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_acados_nmpc_solver.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/acados_codegen_common.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_acados_env.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/acados_ocp_fairino_nmpc.json"
    COMMENT "Generating acados NMPC solver sources into c_generated_code_nmpc/"
    VERBATIM
)

add_custom_target(generate_acados_nmpc_solver_code DEPENDS ${ACADOS_NMPC_GENERATED_FILES})

include_directories(
    include
    ${ACADOS_DIR}/include
    ${ACADOS_DIR}/include/acados
    ${ACADOS_DIR}/include/blasfeo/include
    ${ACADOS_DIR}/include/hpipm/include
    ${ACADOS_GEN_ROOT}
    ${ACADOS_GEN_MODEL_DIR}
    ${ACADOS_GEN_COST_DIR}
    ${ACADOS_GEN_CONSTR_DIR}
    ${ACADOS_NMPC_GEN_ROOT}
    ${ACADOS_NMPC_GEN_MODEL_DIR}
    ${ACADOS_NMPC_GEN_COST_DIR}
    ${ACADOS_NMPC_GEN_CONSTR_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

link_directories(${ACADOS_DIR}/lib)

set(ACADOS_SRC
    "${ACADOS_GEN_ROOT}/acados_solver_fairino_arm_mpc.c"
    "${ACADOS_GEN_MODEL_DIR}/fairino_arm_mpc_expl_ode_fun.c"
    "${ACADOS_GEN_MODEL_DIR}/fairino_arm_mpc_expl_vde_adj.c"
    "${ACADOS_GEN_MODEL_DIR}/fairino_arm_mpc_expl_vde_forw.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_0_fun.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_0_fun_jac.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_0_fun_jac_hess.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_fun.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_fun_jac.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_fun_jac_hess.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_e_fun.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_e_fun_jac.c"
    "${ACADOS_GEN_COST_DIR}/fairino_arm_mpc_cost_ext_cost_e_fun_jac_hess.c"
    "${ACADOS_GEN_CONSTR_DIR}/fairino_arm_mpc_constr_h_fun.c"
    "${ACADOS_GEN_CONSTR_DIR}/fairino_arm_mpc_constr_h_fun_jac_uxt_zt.c"
)

set(ACADOS_NMPC_SRC
    "${ACADOS_NMPC_GEN_ROOT}/acados_solver_fairino_arm_nmpc.c"
    "${ACADOS_NMPC_GEN_MODEL_DIR}/fairino_arm_nmpc_expl_ode_fun.c"
    "${ACADOS_NMPC_GEN_MODEL_DIR}/fairino_arm_nmpc_expl_vde_adj.c"
    "${ACADOS_NMPC_GEN_MODEL_DIR}/fairino_arm_nmpc_expl_vde_forw.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_0_fun.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_0_fun_jac.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_0_fun_jac_hess.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_fun.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_fun_jac.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_fun_jac_hess.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_e_fun.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_e_fun_jac.c"
    "${ACADOS_NMPC_GEN_COST_DIR}/fairino_arm_nmpc_cost_ext_cost_e_fun_jac_hess.c"
    "${ACADOS_NMPC_GEN_CONSTR_DIR}/fairino_arm_nmpc_constr_h_fun.c"
    "${ACADOS_NMPC_GEN_CONSTR_DIR}/fairino_arm_nmpc_constr_h_fun_jac_uxt_zt.c"
)

# 核心执行链源文件（runtime_core）
set(CORE_SRCS
    src/mpc_solver.cpp
    src/nmpc_solver.cpp
    src/solver_selector.cpp
    src/obstacle_distance_ops.cpp
    src/obstacle_tracker.cpp
    src/arc_path_follower.cpp
    src/smooth_box_distance.cpp
    src/robot_kinematics.cpp
    src/rviz_visualizer.cpp
    src/control/control_coordinator.cpp
    src/control/deadlock_replan_engine.cpp
    src/control/command_pipeline.cpp
    src/runtime/mpc_runtime_state.cpp
    src/control/scene_obstacle_provider.cpp
    ${ACADOS_SRC}
    ${ACADOS_NMPC_SRC}
)

set(CORE_DEPS
    rclcpp
    std_msgs
    geometry_msgs
    sensor_msgs
    trajectory_msgs
    moveit_core
    moveit_ros_planning
    tf2_ros
    tf2_eigen
    myrobot_planning_core
    yaml-cpp
    ament_index_cpp
)

# 主节点
add_executable(mpc_avoidance_node
    src/mpc_avoidance_node.cpp
    ${CORE_SRCS}
)
add_dependencies(mpc_avoidance_node generate_acados_solver_code)
add_dependencies(mpc_avoidance_node generate_acados_nmpc_solver_code)

target_link_libraries(mpc_avoidance_node
    acados
    hpipm
    blasfeo
    m
    yaml-cpp
)

# Prefer this package's pinned acados ABI over inherited loader paths.
target_link_options(mpc_avoidance_node PRIVATE "-Wl,--disable-new-dtags")
set_target_properties(mpc_avoidance_node PROPERTIES
    BUILD_RPATH "${ACADOS_DIR}/lib"
    INSTALL_RPATH "${ACADOS_DIR}/lib"
)

ament_target_dependencies(mpc_avoidance_node ${CORE_DEPS})

# 障碍物模拟器节点（可选）
if(BUILD_MPC_SIM_TOOLS)
    add_executable(obstacle_simulator
        src/sim/obstacle_simulator.cpp
    )

    target_include_directories(obstacle_simulator PUBLIC
        ${IGNITION_TRANSPORT_INCLUDE_DIRS}
    )

    target_link_libraries(obstacle_simulator
        ${IGNITION_TRANSPORT_LIBRARIES}
        yaml-cpp
    )

    ament_target_dependencies(obstacle_simulator
        rclcpp
        geometry_msgs
        visualization_msgs
        moveit_msgs
        shape_msgs
        tf2_ros
        yaml-cpp
        ament_index_cpp
    )
endif()

# 安装
install(TARGETS mpc_avoidance_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_MPC_SIM_TOOLS)
    install(TARGETS obstacle_simulator
        RUNTIME DESTINATION lib/${PROJECT_NAME}
    )
endif()

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY config rviz
    DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_deadlock_replan_engine
        test/test_deadlock_replan_engine.cpp
        src/control/deadlock_replan_engine.cpp
        src/runtime/mpc_runtime_state.cpp
    )
    ament_target_dependencies(test_deadlock_replan_engine
        rclcpp
        myrobot_planning_core
    )
endif()

ament_package()
