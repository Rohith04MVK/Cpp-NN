cmake_minimum_required(VERSION 3.0)
project(Cpp-NN CXX)
include_directories(src)

set(CMAKE_CXX_STANDARD 11)
option(CPP_NN_BUILD_EXAMPLE "Whether to build examples" ON)

file(GLOB_RECURSE nn_sources src/*.h)
add_library(Cpp-NN ${nn_sources})
set_target_properties(Cpp-NN PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(Cpp-NN PUBLIC src)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

target_include_directories(Cpp-NN PUBLIC ${EIGEN3_INCLUDE_DIR})


if (CPP_NN_BUILD_EXAMPLE)

    add_executable(iris_test examples/IrisTest.cpp)
    target_link_libraries(iris_test Cpp-NN)
endif()