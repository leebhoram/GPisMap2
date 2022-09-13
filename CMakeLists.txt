cmake_minimum_required(VERSION 3.4)
project(GPisMap2 VERSION 0.1)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set(CMAKE_CXX_STANDARD 14)

file(GLOB_RECURSE SOURCES gpismap/src/**.cpp gpismap/runtime_api.cc)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -Wall -std=c++14 -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-reorder -Wno-sign-compare -Wno-missing-braces")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -DDEBUG")

add_library(gpismap SHARED ${SOURCES})
target_include_directories(gpismap PUBLIC ./gpismap/include ./gpismap)
target_link_libraries(gpismap Eigen3::Eigen )
