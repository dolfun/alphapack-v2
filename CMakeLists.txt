cmake_minimum_required(VERSION 3.20)
project(alphapack)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(Deps.cmake)
add_subdirectory(src)
