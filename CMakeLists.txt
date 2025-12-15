cmake_minimum_required(VERSION 3.20)
project(alphapack)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(Deps.cmake)

add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_23)
target_compile_options(
  project_options
  INTERFACE $<$<CXX_COMPILER_ID:MSVC>:
            /W4
            /WX>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
            -Wall
            -Wextra
            -Wpedantic
            -Werror>
)
target_include_directories(
  project_options INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

add_subdirectory(src)
