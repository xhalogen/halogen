cmake_minimum_required(VERSION 3.20)
project(
    Halogen
    VERSION 0.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(Halogen main.cpp)
add_executable(TensorTests tests/TensorTests.cpp)

target_include_directories(Halogen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(TensorTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
