cmake_minimum_required(VERSION 3.16)
project(lab2 C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")


add_executable(lab2 main.c)
target_link_libraries(lab2 m)
