add_library(
  memory
  INTERFACE
  allocator.h
  storage_buffer.h
  trivial_object_pool.h
)

target_link_libraries(memory INTERFACE project_options)
