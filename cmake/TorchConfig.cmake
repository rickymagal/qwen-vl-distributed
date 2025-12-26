if (NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found. Set TORCH_DIR to LibTorch root.")
endif()

set(TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
