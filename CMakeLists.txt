cmake_minimum_required(VERSION 3.18)
project(qwen_vl_distributed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA-only
find_package(CUDA REQUIRED)

# LibTorch
list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
find_package(Torch REQUIRED)

include(cmake/TorchConfig.cmake)

include_directories(include)

add_compile_options(-O3)

# Core library
file(GLOB_RECURSE CORE_SRC
    src/core/*.cpp
    src/vision/*.cpp
    src/model/*.cpp
    src/loader/*.cpp
    src/runtime/*.cpp
)

add_library(qwen_core STATIC ${CORE_SRC})
target_link_libraries(qwen_core "${TORCH_LIBRARIES}")
target_compile_definitions(qwen_core PRIVATE USE_CUDA)

# Pipeline stages
file(GLOB STAGE_DIRS stages/*)

foreach(stage ${STAGE_DIRS})
    if(IS_DIRECTORY ${stage})
        get_filename_component(stage_name ${stage} NAME)
        add_executable(${stage_name} ${stage}/main.cpp)
        target_link_libraries(${stage_name} qwen_core "${TORCH_LIBRARIES}")
    endif()
endforeach()

set_property(TARGET qwen_core PROPERTY CXX_STANDARD 17)
