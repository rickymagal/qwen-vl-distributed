#include "runtime/transport.h"

#include <torch/torch.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

int main() {
  std::unique_ptr<qwen::TcpServer> server;
  try {
    server = std::make_unique<qwen::TcpServer>(0);
  } catch (const std::exception& e) {
    std::string msg = e.what();
    if (msg.find("Operation not permitted") != std::string::npos ||
        msg.find("permission") != std::string::npos) {
      std::fprintf(stderr, "SKIP: %s\n", msg.c_str());
      return 0;
    }
    std::fprintf(stderr, "transport init error: %s\n", msg.c_str());
    return 1;
  }
  const int port = server->port();

  std::mutex mu;
  std::string err;
  qwen::ActivationPacket recv_act;
  qwen::KVPacket recv_kv;

  std::thread t([&]() {
    try {
      qwen::TcpConn conn(server->accept_one());
      recv_act = conn.recv_activation();
      recv_kv = conn.recv_kv();
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(mu);
      err = e.what();
    }
  });

  qwen::TcpClient client("127.0.0.1", port);
  auto hidden = torch::arange(0, 6, torch::TensorOptions().dtype(torch::kFloat32)).view({1, 2, 3});
  auto mask = torch::tensor({{1.0f, 0.0f}}, torch::TensorOptions().dtype(torch::kFloat32));
  qwen::ActivationPacket send_act;
  send_act.stage_from = 1;
  send_act.stage_to = 2;
  send_act.step = 7;
  send_act.pos = 13;
  send_act.hidden = hidden;
  send_act.attn_mask = mask;
  client.send_activation(send_act);

  auto k = torch::arange(0, 2 * 1 * 2 * 3 * 4,
                         torch::TensorOptions().dtype(torch::kFloat32))
               .view({2, 1, 2, 3, 4});
  auto v = k + 1.0f;
  qwen::KVPacket send_kv;
  send_kv.stage_from = 1;
  send_kv.stage_to = 2;
  send_kv.step = 7;
  send_kv.pos = 13;
  send_kv.k = k;
  send_kv.v = v;
  client.send_kv(send_kv);

  t.join();

  if (!err.empty()) {
    std::fprintf(stderr, "transport error: %s\n", err.c_str());
    return 1;
  }

  if (recv_act.stage_from != send_act.stage_from || recv_act.stage_to != send_act.stage_to ||
      recv_act.step != send_act.step || recv_act.pos != send_act.pos) {
    std::fprintf(stderr, "activation metadata mismatch\n");
    return 1;
  }
  if (!recv_act.hidden.defined() || !torch::equal(recv_act.hidden, hidden)) {
    std::fprintf(stderr, "activation hidden mismatch\n");
    return 1;
  }
  if (!recv_act.attn_mask.has_value() || !torch::equal(recv_act.attn_mask.value(), mask)) {
    std::fprintf(stderr, "activation attn_mask mismatch\n");
    return 1;
  }

  if (recv_kv.stage_from != send_kv.stage_from || recv_kv.stage_to != send_kv.stage_to ||
      recv_kv.step != send_kv.step || recv_kv.pos != send_kv.pos) {
    std::fprintf(stderr, "kv metadata mismatch\n");
    return 1;
  }
  if (!recv_kv.k.has_value() || !recv_kv.v.has_value()) {
    std::fprintf(stderr, "kv missing tensors\n");
    return 1;
  }
  if (!torch::equal(recv_kv.k.value(), k) || !torch::equal(recv_kv.v.value(), v)) {
    std::fprintf(stderr, "kv tensor mismatch\n");
    return 1;
  }

  return 0;
}
