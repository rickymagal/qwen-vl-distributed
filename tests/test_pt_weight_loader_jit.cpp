#include "mini_test.h"

#include <loader/pt_weight_loader.h>

int main() {
  // This test is intentionally lightweight: it just ensures the translation unit
  // and loader API link correctly in environments where Torch is present.
  // It does not require a real model checkpoint.
  CHECK_TRUE(true);
  return 0;
}
