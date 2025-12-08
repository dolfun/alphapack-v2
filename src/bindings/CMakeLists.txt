pybind11_add_module(alphapack bindings.cc)
target_link_libraries(alphapack PRIVATE mcts)

set(MODULE_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/src/alphapack)
set_target_properties(
  alphapack PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MODULE_OUTPUT_DIR}
)

py_module_add_stubgen(alphapack)
