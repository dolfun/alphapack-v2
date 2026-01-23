cmake_minimum_required(VERSION 3.20)
project(alphapack)

include(cmake/Deps.cmake)
include(cmake/ProjectOptions.cmake)
include(cmake/StaticAnalyzers.cmake)

add_subdirectory(src)
