#include "mini_test.h"

#include <core/kv_cache.h>

int main() {
  // Minimal construction smoke test.
  qwen::KVCache cache;
  CHECK_TRUE(true);
  (void)cache;
  return 0;
}
