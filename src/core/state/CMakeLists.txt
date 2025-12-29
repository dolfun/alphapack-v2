add_library(
  state
  ndarray.h
  item.h
  state.h
  state.cc
  serializer.h
  state_factory.h
  state_factory.cc
)

target_link_libraries(state PRIVATE project_options)
