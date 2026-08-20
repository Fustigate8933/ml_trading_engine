cmake_minimum_required(VERSION 3.28)

project(ml_trading_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -Wall -Wextra")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")

# use release build if not specified in cmd line arg
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Generate compile_commands.json for LSP (clangd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(engine
    src/main.cpp
)

target_link_libraries(engine PRIVATE CUDA::cudart)

set_target_properties(engine PROPERTIES
    CUDA_ARCHITECTURES native
)

