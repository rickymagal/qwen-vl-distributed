#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace qwen_test {

inline void fail(const char* file, int line, const std::string& msg) {
  std::fprintf(stderr, "TEST FAIL %s:%d: %s\n", file, line, msg.c_str());
  std::fflush(stderr);
  std::exit(1);
}

#define QT_ASSERT_TRUE(expr) \
  do { \
    if (!(expr)) { \
      ::qwen_test::fail(__FILE__, __LINE__, std::string("assertion failed: ") + #expr); \
    } \
  } while (0)

#define QT_ASSERT_EQ(a, b) \
  do { \
    auto _a = (a); \
    auto _b = (b); \
    if (!(_a == _b)) { \
      ::qwen_test::fail(__FILE__, __LINE__, std::string("assertion failed: ") + #a + " == " + #b); \
    } \
  } while (0)

inline bool cuda_available_or_skip() {
  if (!torch::cuda::is_available()) {
    std::printf("skipped: cuda not available\n");
    return false;
  }
  return true;
}

} // namespace qwen_test
