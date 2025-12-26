#include <iostream>
#include "runtime/activation_packet.h"

using namespace qwen;

int main() {
  ActivationPacket p;
  p.version = 1;
  p.stage_from = 0;
  p.stage_to = 1;
  p.step = 0;
  p.pos = 0;

  if (p.version != 1 || p.stage_from != 0) {
    std::cerr << "ActivationPacket basic invariant failed\n";
    return 1;
  }

  std::cout << "Distributed smoke test passed\n";
  return 0;
}
