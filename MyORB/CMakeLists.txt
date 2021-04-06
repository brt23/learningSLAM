cmake_minimum_required(VERSION 2.8.5)
project(MyVO)

add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

set(OpenCV_DIR /home/y/文档/Code/learningSLAM/3rdparty/opencv4/lib/cmake)
find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_include_DIRS}
)

add_executable(${PROJECT_NAME} src/self_ORB.cpp)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)