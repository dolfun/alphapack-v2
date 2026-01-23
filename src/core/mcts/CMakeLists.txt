add_library(mcts model_adapter.cc model_adapter.h)
target_link_libraries(mcts PRIVATE memory)
enable_project_options(mcts)
