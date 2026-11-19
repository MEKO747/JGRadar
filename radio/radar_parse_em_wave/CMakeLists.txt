cmake_minimum_required(VERSION 3.10)
project(radar_parse_em_wave)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
option(RADAR_BUILD_FLOWGRAPH "Generate and install the GNU Radio flowgraph" ON)

ament_python_install_package(rmuc)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RadarParseEmWaveDemodConfig.msg"
  "msg/RadarParseEmWave0A01RobotPosition.msg"
  "msg/RadarParseEmWave0A02RobotHp.msg"
  "msg/RadarParseEmWave0A03RobotAmmo.msg"
  "msg/RadarParseEmWave0A04FieldStatus.msg"
  "msg/RadarParseEmWave0A05RobotBuff.msg"
  "msg/RadarParseEmWave0A06InterferenceKey.msg"
  DEPENDENCIES std_msgs
)

if(RADAR_BUILD_FLOWGRAPH)
  find_program(GRCC_EXECUTABLE grcc)
  if(NOT GRCC_EXECUTABLE)
    message(FATAL_ERROR "GNU Radio grcc is required. Install the gnuradio package.")
  endif()

  set(FLOWGRAPH_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/grc/RMUC2026_NanoSDR_RX.grc")
  set(FLOWGRAPH_PYTHON "${CMAKE_CURRENT_BINARY_DIR}/RMUC2026_NanoSDR_RX.py")
  add_custom_command(
    OUTPUT "${FLOWGRAPH_PYTHON}"
    COMMAND "${GRCC_EXECUTABLE}" -o "${CMAKE_CURRENT_BINARY_DIR}" "${FLOWGRAPH_SOURCE}"
    DEPENDS "${FLOWGRAPH_SOURCE}"
    VERBATIM
  )
  add_custom_target(generate_radio_flowgraph ALL DEPENDS "${FLOWGRAPH_PYTHON}")
  install(FILES "${FLOWGRAPH_SOURCE}" "${FLOWGRAPH_PYTHON}" DESTINATION share/${PROJECT_NAME}/grc)
endif()

install(
  PROGRAMS scripts/rmuc_ros2_node.py
  RENAME radar_parse_em_wave_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
