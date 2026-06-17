cmake_minimum_required(VERSION 3.16)
project(roboparty-inference)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Boost COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(roboparty_motors REQUIRED)
find_package(roboparty_imu REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

option(YAML_BUILD_SHARED_LIBS "Build shared library" OFF)
option(YAML_CPP_BUILD_TESTS "Build tests" OFF)
option(YAML_CPP_BUILD_TOOLS "Build tools" OFF)
add_subdirectory(thirdparty)

set(PUBLIC_DEPENDENCIES
    ${Boost_LIBRARIES} pthread Eigen3::Eigen fmt::fmt spdlog::spdlog ${ONNXRUNTIME_LIB_FILE} yaml-cpp)

add_subdirectory(src)

add_library(robot STATIC
  src/robot_interface.cpp
)

target_include_directories(robot
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(robot PUBLIC ${PUBLIC_DEPENDENCIES} utils cnpy
  roboparty_motors::roboparty_motors
  roboparty_imu::roboparty_imu
)

pybind11_add_module(robot_py src/pybind_module.cpp)
target_link_libraries(robot_py PUBLIC robot)

add_executable(${PROJECT_NAME}_node src/inference_node.cpp src/ros_interface.cpp src/obs_manager.cpp)
target_include_directories(${PROJECT_NAME}_node
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CNPY_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME}_node PUBLIC ${PUBLIC_DEPENDENCIES} utils cnpy robot)
ament_target_dependencies(${PROJECT_NAME}_node PUBLIC rclcpp sensor_msgs geometry_msgs std_srvs)

set_target_properties(${PROJECT_NAME}_node PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(TARGETS robot utils
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)
install(FILES ${ONNXRUNTIME_LIB_FILE} ${ONNXRUNTIME_LIB_FILE}.1 ${ONNXRUNTIME_LIB_FILE}.1.21.0
        DESTINATION lib)

set(PYTHON_INSTALL_DIR "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

install(TARGETS robot_py
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
  ARCHIVE DESTINATION ${PYTHON_INSTALL_DIR}
  RUNTIME DESTINATION ${PYTHON_INSTALL_DIR})

ament_export_dependencies(rclcpp sensor_msgs geometry_msgs)
ament_export_libraries(robot)

ament_package()
