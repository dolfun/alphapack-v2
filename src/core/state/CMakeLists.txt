add_library(
  state
  item.h
  ndarray.h
  serializer.h
  state.cc
  state.h
  state_factory.cc
  state_factory.h
)

target_link_libraries(state PRIVATE project_options)
