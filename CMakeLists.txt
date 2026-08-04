cmake_minimum_required(VERSION 3.15)
project(zerocross_engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release to ensure -O3 optimization on Kaggle
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Option to enable sanitizers (default OFF for safe Kaggle builds)
option(BUILD_WITH_SANITIZERS "Build tests with ASan and UBSan" OFF)

add_executable(test_game_state EXCLUDE_FROM_ALL
    src/game_state.cpp 
    tests/cpp/test_game_state.cpp
)
target_include_directories(test_game_state PRIVATE include)
target_compile_options(test_game_state PRIVATE -g)

add_executable(test_mcts EXCLUDE_FROM_ALL
    src/game_state.cpp 
    src/mcts.cpp 
    tests/cpp/test_mcts.cpp
)
target_include_directories(test_mcts PRIVATE include)
target_compile_options(test_mcts PRIVATE -g)

add_executable(test_stress EXCLUDE_FROM_ALL
    src/game_state.cpp 
    tests/cpp/test_stress.cpp
)
target_include_directories(test_stress PRIVATE include)
target_compile_options(test_stress PRIVATE -O3 -g)

# Apply sanitizers only if explicitly enabled
if(BUILD_WITH_SANITIZERS)
    target_compile_options(test_game_state PRIVATE -fsanitize=address -fsanitize=undefined)
    target_link_options(test_game_state PRIVATE -fsanitize=address -fsanitize=undefined)
    
    target_compile_options(test_mcts PRIVATE -fsanitize=address -fsanitize=undefined)
    target_link_options(test_mcts PRIVATE -fsanitize=address -fsanitize=undefined)
    
    target_compile_options(test_stress PRIVATE -fsanitize=address -fsanitize=undefined)
    target_link_options(test_stress PRIVATE -fsanitize=address -fsanitize=undefined)
endif()

if(EMSCRIPTEN)
    # WebAssembly Build Configuration
    message(STATUS "Emscripten detected: Building WebAssembly module...")
    
    add_executable(zerocross_wasm
        src/wasm_bindings.cpp
        src/game_state.cpp
        src/mcts.cpp
    )
    
    target_include_directories(zerocross_wasm PRIVATE include)
    target_compile_options(zerocross_wasm PRIVATE -O3)
    
    # Embind flags and memory configurations (No spaces after -s to prevent CMake parsing bugs)
    set_target_properties(zerocross_wasm PROPERTIES OUTPUT_NAME "zerocross")
    target_link_options(zerocross_wasm PRIVATE
        --bind
        -sALLOW_MEMORY_GROWTH=1
        -sMODULARIZE=1
        -sEXPORT_NAME=createZeroCrossModule
        -sENVIRONMENT=web
    )
else()
    # Native Python Build Configuration
    message(STATUS "Native build detected: Building pybind11 module...")
    
    # Automatically discover pybind11's CMake directory from the active Python environment
    execute_process(
        COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE pybind11_status
    )

    if(NOT pybind11_status EQUAL 0)
        message(FATAL_ERROR "pybind11 not found! Hint: Run 'python -m pip install pybind11' in your active Python environment before building.")
    endif()

    find_package(pybind11 REQUIRED)

    pybind11_add_module(zerocross_engine
        src/bindings.cpp
        src/game_state.cpp
        src/mcts.cpp
    )
    target_include_directories(zerocross_engine PRIVATE include)
    target_compile_options(zerocross_engine PRIVATE -O3 -fPIC)
endif()