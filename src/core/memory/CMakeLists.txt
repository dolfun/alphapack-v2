add_library(
  memory
  INTERFACE
  alloc_fn_utils.h
  memory_block_pool.h
  raw_buffer.h
)

target_link_libraries(memory INTERFACE project_options)
