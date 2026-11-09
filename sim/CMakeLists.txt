cmake_minimum_required(VERSION 3.20)
project(npusim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

add_executable(npusim src/main.cpp src/npusim.cpp)
target_include_directories(npusim PRIVATE include)
