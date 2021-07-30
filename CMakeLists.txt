cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(SEMANTIC_ICP)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Eigen3 3.3.0 REQUIRED)
find_package(Sophus REQUIRED)
find_package (Ceres REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Sophus_INCLUDE_DIR}
    ${CERES_INCLUDES}
)


add_executable (lcr-smm lcr-smm.cc)
target_link_libraries (lcr-smm ${PCL_LIBRARIES} ${CERES_LIBRARIES})
