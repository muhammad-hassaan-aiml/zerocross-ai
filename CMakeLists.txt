cmake_minimum_required(VERSION 3.15)
project(zerocross_engine)

# Enforce C++17 standard for modern language features and standard library components
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Debug build to ensure symbols and sanitizers are active during development
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Compile core C++ game state and unit tests.
# TODO: Integrate pybind11 target for Python bindings once core engine is validated.
add_executable(test_game_state 
    src/game_state.cpp 
    tests/cpp/test_game_state.cpp
)

target_include_directories(test_game_state PRIVATE include)

# Enable Address (ASan) and Undefined Behavior (UBSan) sanitizers for GCC/Clang
# -g generates debugging information for stack traces
target_compile_options(test_game_state PRIVATE -fsanitize=address -fsanitize=undefined -g)
target_link_options(test_game_state PRIVATE -fsanitize=address -fsanitize=undefined)