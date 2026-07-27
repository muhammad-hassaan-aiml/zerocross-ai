cmake_minimum_required(VERSION 3.15)
project(zerocross_engine)

# Enforce C++17 standard for modern language features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Core Game State Target
add_executable(test_game_state 
    src/game_state.cpp 
    tests/cpp/test_game_state.cpp
)
target_include_directories(test_game_state PRIVATE include)
target_compile_options(test_game_state PRIVATE -fsanitize=address -fsanitize=undefined -g)
target_link_options(test_game_state PRIVATE -fsanitize=address -fsanitize=undefined)

# Monte Carlo Tree Search Target
add_executable(test_mcts 
    src/game_state.cpp 
    src/mcts.cpp 
    tests/cpp/test_mcts.cpp
)
target_include_directories(test_mcts PRIVATE include)
target_compile_options(test_mcts PRIVATE -fsanitize=address -fsanitize=undefined -g)
target_link_options(test_mcts PRIVATE -fsanitize=address -fsanitize=undefined)

find_package(pybind11 REQUIRED)

pybind11_add_module(zerocross_engine
    src/bindings.cpp
    src/game_state.cpp
    src/mcts.cpp
)
target_include_directories(zerocross_engine PRIVATE include)
target_compile_options(zerocross_engine PRIVATE -O3 -fPIC)