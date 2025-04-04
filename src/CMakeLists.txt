cmake_minimum_required(VERSION 3.10.0)

# 添加头文件目录
include_directories(${CMAKE_SOURCE_DIR}/src)

find_package(OpenCV REQUIRED)

add_executable(cv_basic_learn main.cpp quick.cpp)

# 链接OpenCV库到你的项目
target_link_libraries(cv_basic_learn ${OpenCV_LIBS})
# target_link_libraries(cv_basic_learn quick.cpp quick.h)
