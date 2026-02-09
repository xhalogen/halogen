cmake_minimum_required(VERSION 3.20)

project(
    Halogen
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# absalg
add_library(absalg INTERFACE)
target_include_directories(absalg INTERFACE include)

# excutables
add_executable(absalg_test test/test.cpp)
target_link_libraries(absalg_test PRIVATE absalg)
