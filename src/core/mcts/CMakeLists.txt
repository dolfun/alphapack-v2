add_library(mcts INTERFACE model_info.h)

target_link_libraries(mcts INTERFACE project_options)
