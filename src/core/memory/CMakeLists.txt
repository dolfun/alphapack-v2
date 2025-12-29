add_library(
  memory
  INTERFACE
  allocator.h
  storage_buffer.h
  object_pool.h
  batched_array_pool.h
)

target_link_libraries(memory INTERFACE project_options)
