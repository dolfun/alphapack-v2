add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_23)
target_compile_options(
  project_options
  INTERFACE $<$<CXX_COMPILER_ID:MSVC>:
            /W4
            /WX
            /wd4324
            $<$<CONFIG:Release>:
            /fp:fast
            /arch:AVX2
            >
            >
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            $<$<CONFIG:Release>:
            -ffast-math
            -march=raptorlake
            >
            >
)

target_include_directories(
  project_options INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

set_target_properties(
  project_options PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE
)
