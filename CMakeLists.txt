cmake_minimum_required(VERSION 3.18)
project(qwen_vl_distributed LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA toolkit (no deprecated FindCUDA)
find_package(CUDAToolkit REQUIRED)

# LibTorch
if (DEFINED Torch_DIR)
  find_package(Torch REQUIRED)
else()
  if (DEFINED TORCH_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
  endif()
  find_package(Torch REQUIRED)
endif()

include(cmake/TorchConfig.cmake)

add_compile_options(-O3)

# Core sources
file(GLOB_RECURSE CORE_SRC
  src/core/*.cpp
  src/vision/*.cpp
  src/model/*.cpp
  src/loader/*.cpp
  src/runtime/*.cpp
)

add_library(qwen_core STATIC ${CORE_SRC})
target_include_directories(qwen_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(qwen_core PUBLIC "${TORCH_LIBRARIES}" CUDA::cudart)
target_compile_definitions(qwen_core PRIVATE USE_CUDA)

# Some distros need pthread explicitly for networking / std::thread usage.
find_package(Threads REQUIRED)
target_link_libraries(qwen_core PUBLIC Threads::Threads)

set_property(TARGET qwen_core PROPERTY CXX_STANDARD 17)

# Stage binaries: stages/<name>/main.cpp -> build/<name>
file(GLOB STAGE_DIRS stages/*)

foreach(stage ${STAGE_DIRS})
  if (IS_DIRECTORY ${stage})
    get_filename_component(stage_name ${stage} NAME)
    add_executable(${stage_name} ${stage}/main.cpp)
    target_include_directories(${stage_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${stage_name} PRIVATE qwen_core "${TORCH_LIBRARIES}" CUDA::cudart Threads::Threads)
    set_property(TARGET ${stage_name} PROPERTY CXX_STANDARD 17)
  endif()
endforeach()

# Smoke test (CUDA translation unit)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_smoke_forward.cu")
  add_executable(test_smoke_forward_cuda tests/test_smoke_forward.cu)
  target_include_directories(test_smoke_forward_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_smoke_forward_cuda PRIVATE qwen_core "${TORCH_LIBRARIES}" CUDA::cudart Threads::Threads)
  set_property(TARGET test_smoke_forward_cuda PROPERTY CXX_STANDARD 17)
endif()
