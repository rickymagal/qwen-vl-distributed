# Unit tests for qwen_core.
# These tests are designed to run without any real Qwen weights/model files.

function(qwen_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE qwen_core)
  add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
endfunction()

qwen_add_test(test_tensor_utils
  test_tensor_utils.cpp
)

qwen_add_test(test_pt_weight_loader_jit
  test_pt_weight_loader_jit.cpp
)

qwen_add_test(test_embedding_cuda
  test_embedding_cuda.cpp
)

qwen_add_test(test_rope_cuda
  test_rope_cuda.cpp
)

qwen_add_test(test_kv_cache_cuda
  test_kv_cache_cuda.cpp
)

qwen_add_test(test_attention_cuda
  test_attention_cuda.cpp
)

qwen_add_test(test_smoke_forward_cuda
  test_smoke_forward.cu
)

qwen_add_test(test_model_loader
  test_model_loader.cpp
)

qwen_add_test(test_kv_wire
  test_kv_wire.cpp
)

qwen_add_test(test_transport_kv
  test_transport_kv.cpp
)

add_test(
  NAME test_vision_manifest
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../python_export/validate_vision_manifest.py
          --hf-config ${CMAKE_CURRENT_SOURCE_DIR}/../python_export/reduced_export_out/hf_config.json
          --manifest ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/vision_manifest.json
          --min-blocks 2
)
