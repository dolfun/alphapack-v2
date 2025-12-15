add_library(
  state
  ndarray.h
  item.h
  state.h
  state.cc
  serializer.h
)

target_link_libraries(state PRIVATE project_options)
