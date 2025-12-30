#pragma once

#include "runtime/activation_packet.h"

#include <cstdint>
#include <string>

namespace qwen {

class TcpClient {
public:
  TcpClient(const std::string& host, int port);
  ~TcpClient();

  void send_activation(const ActivationPacket& p);
  ActivationPacket recv_activation();

private:
  int fd_ = -1;
};

class TcpServer {
public:
  explicit TcpServer(int port);
  ~TcpServer();

  int accept_one();

private:
  int fd_ = -1;
};

class TcpConn {
public:
  explicit TcpConn(int fd);
  ~TcpConn();

  void send_activation(const ActivationPacket& p);
  ActivationPacket recv_activation();

  void send_activation_raw(const ActivationPacket& p);

private:
  int fd_ = -1;
};

} // namespace qwen
