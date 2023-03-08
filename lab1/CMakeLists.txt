cmake_minimum_required(VERSION 3.16)
project(lab1 C)

set(CMAKE_C_STANDARD 99)

include_directories(/usr/include/x86_64-linux-gnu/mpich/include)
set(CMAKE_C_COMPILER "/usr/bin/mpicc")
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

add_executable(lab1 main.c)
target_link_libraries(lab1 m)