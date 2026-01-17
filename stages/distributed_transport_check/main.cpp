#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "runtime/activation_packet.h"
#include "runtime/transport.h"

static const char* arg_str(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return argv[i + 1];
  }
  return def;
}

static int64_t arg_i64(int argc, char** argv, const char* key, int64_t def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == key) return std::stoll(argv[i + 1]);
  }
  return def;
}

static void usage() {
  std::fprintf(stderr,
               "distributed_transport_check usage:\n"
               "  --mode <client|server>\n"
               "  --port <port>\n"
               "  [--host <host>]                 (client only)\n"
               "  [--shape <B,T,D>]               (client only, default 1,8,64)\n"
               "  [--dtype <fp16|bf16|fp32>]      (client only, default fp16)\n"
               "  [--seed <n>]                    (client only)\n");
}

static uint64_t checksum_bytes(const torch::Tensor& t) {
  if (!t.defined()) return 0;
  torch::Tensor cpu = t;
  if (cpu.is_cuda()) cpu = cpu.to(torch::kCPU);
  if (!cpu.is_contiguous()) cpu = cpu.contiguous();
  const uint8_t* p = static_cast<const uint8_t*>(cpu.data_ptr());
  const size_t n = (size_t)cpu.nbytes();
  uint64_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += static_cast<uint64_t>(p[i]);
  }
  return sum;
}

static torch::Tensor make_checksum_tensor(uint64_t sum, uint64_t nbytes) {
  auto t = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  t[0] = (int64_t)sum;
  t[1] = (int64_t)nbytes;
  return t;
}

static void parse_shape(const std::string& s, int64_t* b, int64_t* t, int64_t* d) {
  if (!b || !t || !d) return;
  *b = 1;
  *t = 8;
  *d = 64;
  size_t p1 = s.find(',');
  size_t p2 = s.find(',', p1 == std::string::npos ? p1 : p1 + 1);
  if (p1 == std::string::npos || p2 == std::string::npos) return;
  *b = std::stoll(s.substr(0, p1));
  *t = std::stoll(s.substr(p1 + 1, p2 - p1 - 1));
  *d = std::stoll(s.substr(p2 + 1));
}

int main(int argc, char** argv) {
  const std::string mode = arg_str(argc, argv, "--mode", "");
  const int64_t port = arg_i64(argc, argv, "--port", -1);
  if (mode.empty() || port < 0) {
    usage();
    return 2;
  }

  if (mode == "server") {
    qwen::TcpServer server((int)port);
    qwen::TcpConn conn(server.accept_one());
    qwen::ActivationPacket p = conn.recv_activation();
    if (!p.hidden.defined()) {
      std::fprintf(stderr, "error: hidden undefined\n");
      return 3;
    }
    if (!p.attn_mask.has_value() || !p.attn_mask->defined()) {
      std::fprintf(stderr, "error: checksum tensor missing\n");
      return 3;
    }
    auto chk = p.attn_mask->to(torch::kCPU);
    if (chk.numel() != 2 || chk.scalar_type() != torch::kInt64) {
      std::fprintf(stderr, "error: checksum tensor invalid\n");
      return 3;
    }
    const uint64_t sum_expected = (uint64_t)chk[0].item<int64_t>();
    const uint64_t nbytes_expected = (uint64_t)chk[1].item<int64_t>();
    const uint64_t sum = checksum_bytes(p.hidden);
    const uint64_t nbytes = (uint64_t)p.hidden.nbytes();
    if (sum != sum_expected || nbytes != nbytes_expected) {
      std::fprintf(stderr,
                   "checksum mismatch: expected (sum=%llu nbytes=%llu) got (sum=%llu nbytes=%llu)\n",
                   (unsigned long long)sum_expected,
                   (unsigned long long)nbytes_expected,
                   (unsigned long long)sum,
                   (unsigned long long)nbytes);
      return 4;
    }
    std::fprintf(stderr, "checksum ok\n");
    return 0;
  }

  if (mode == "client") {
    const std::string host = arg_str(argc, argv, "--host", "");
    if (host.empty()) {
      usage();
      return 2;
    }

    const std::string shape_s = arg_str(argc, argv, "--shape", "1,8,64");
    const std::string dtype_s = arg_str(argc, argv, "--dtype", "fp16");
    const int64_t seed = arg_i64(argc, argv, "--seed", 1234);

    int64_t B, T, D;
    parse_shape(shape_s, &B, &T, &D);

    torch::manual_seed((uint64_t)seed);
    c10::ScalarType dtype = torch::kFloat16;
    if (dtype_s == "bf16") dtype = torch::kBFloat16;
    if (dtype_s == "fp32") dtype = torch::kFloat32;

    auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    auto hidden = torch::randn({B, T, D}, opts);

    const uint64_t sum = checksum_bytes(hidden);
    const uint64_t nbytes = (uint64_t)hidden.nbytes();

    qwen::ActivationPacket p;
    p.stage_from = 0;
    p.stage_to = 1;
    p.step = 0;
    p.pos = 0;
    p.hidden = hidden;
    p.attn_mask = make_checksum_tensor(sum, nbytes);

    qwen::TcpClient client(host, (int)port);
    client.send_activation(p);
    std::fprintf(stderr, "sent checksum sum=%llu nbytes=%llu\n",
                 (unsigned long long)sum,
                 (unsigned long long)nbytes);
    return 0;
  }

  usage();
  return 2;
}
