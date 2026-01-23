find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
message(STATUS "Found clang-tidy: ${CLANG_TIDY}")
set(CLANG_TIDY_OPTIONS
    ${CLANG_TIDY} -extra-arg=-Wno-unknown-warning-option
    -extra-arg=-Wno-ignored-optimization-argument
    -extra-arg=-Wno-unused-command-line-argument
)
if(MSVC)
  list(APPEND CLANG_TIDY_OPTIONS --extra-arg=/EHsc)
endif()

find_program(CLANG_FORMAT NAMES clang-format REQUIRED)
message(STATUS "Found clang-format: ${CLANG_FORMAT}")
add_custom_target(format)

function(enable_project_options target)
  get_target_property(TARGET_TYPE ${target} TYPE)
  if(TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
    set(VISIBILITY INTERFACE)
  else()
    set(VISIBILITY PRIVATE)
  endif()
  target_link_libraries(${target} ${VISIBILITY} project_options)

  set(options NOLINT)
  set(oneValueArgs "")
  set(multiValueArgs "")
  cmake_parse_arguments(
    ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )
  if(NOT ARG_NOLINT AND NOT TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
    set_target_properties(
      ${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_OPTIONS}"
    )
  endif()

  get_target_property(TARGET_SOURCES ${target} SOURCES)
  if(TARGET_SOURCES)
    add_custom_target(
      ${target}_format
      COMMAND ${CLANG_FORMAT} -i ${TARGET_SOURCES}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND_EXPAND_LISTS
    )
    add_dependencies(format ${target}_format)
  endif()
endfunction()
