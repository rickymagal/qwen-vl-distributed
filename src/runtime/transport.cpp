#include "runtime/transport.h"

#include <torch/torch.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen {

static void throw_sys(const std::string& msg) {
  throw std::runtime_error(msg + ": " + std::string(std::strerror(errno)));
}

static void write_all(int fd, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  while (n) {
    ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
    if (w < 0) {
      if (errno == EINTR) continue;
      throw_sys("send");
    }
    p += (size_t)w;
    n -= (size_t)w;
  }
}

static void read_all(int fd, void* data, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(data);
  while (n) {
    ssize_t r = ::recv(fd, p, n, MSG_WAITALL);
    if (r <= 0) {
      if (r < 0 && errno == EINTR) continue;
      throw_sys("recv");
    }
    p += (size_t)r;
    n -= (size_t)r;
  }
}

static uint64_t hton_u64(uint64_t x) {
  uint32_t hi = htonl((uint32_t)(x >> 32));
  uint32_t lo = htonl((uint32_t)(x & 0xffffffffu));
  return ((uint64_t)lo << 32) | hi;
}

static uint64_t ntoh_u64(uint64_t x) {
  uint32_t lo = ntohl((uint32_t)(x >> 32));
  uint32_t hi = ntohl((uint32_t)(x & 0xffffffffu));
  return ((uint64_t)hi << 32) | lo;
}

static int32_t scalar_type_to_i32(c10::ScalarType t) {
  return (int32_t)t;
}

static c10::ScalarType i32_to_scalar_type(int32_t v) {
  return (c10::ScalarType)v;
}

static void send_tensor(int fd, const torch::Tensor& t) {
  if (!t.defined()) {
    uint8_t defined = 0;
    write_all(fd, &defined, 1);
    return;
  }

  uint8_t defined = 1;
  write_all(fd, &defined, 1);

  // Serialize as CPU contiguous bytes.
  torch::Tensor cpu = t;
  if (cpu.is_cuda()) cpu = cpu.to(torch::kCPU);
  if (!cpu.is_contiguous()) cpu = cpu.contiguous();

  const int32_t dtype_i = scalar_type_to_i32(cpu.scalar_type());
  const int32_t ndim = (int32_t)cpu.dim();

  int32_t dtype_net = htonl((uint32_t)dtype_i);
  int32_t ndim_net  = htonl((uint32_t)ndim);

  write_all(fd, &dtype_net, sizeof(dtype_net));
  write_all(fd, &ndim_net, sizeof(ndim_net));

  std::vector<int64_t> sizes;
  sizes.reserve((size_t)ndim);
  for (int i = 0; i < ndim; ++i) sizes.push_back(cpu.size(i));

  for (int i = 0; i < ndim; ++i) {
    uint64_t s = hton_u64((uint64_t)sizes[(size_t)i]);
    write_all(fd, &s, sizeof(s));
  }

  uint64_t nbytes = (uint64_t)cpu.nbytes();
  uint64_t nbytes_net = hton_u64(nbytes);
  write_all(fd, &nbytes_net, sizeof(nbytes_net));

  write_all(fd, cpu.data_ptr(), (size_t)nbytes);
}

static torch::Tensor recv_tensor(int fd) {
  uint8_t defined = 0;
  read_all(fd, &defined, 1);
  if (!defined) return torch::Tensor();

  int32_t dtype_net = 0, ndim_net = 0;
  read_all(fd, &dtype_net, sizeof(dtype_net));
  read_all(fd, &ndim_net, sizeof(ndim_net));

  int32_t dtype_i = (int32_t)ntohl((uint32_t)dtype_net);
  int32_t ndim    = (int32_t)ntohl((uint32_t)ndim_net);

  if (ndim < 0 || ndim > 16) {
    throw std::runtime_error("recv_tensor: invalid ndim");
  }

  std::vector<int64_t> sizes((size_t)ndim);
  for (int i = 0; i < ndim; ++i) {
    uint64_t s_net = 0;
    read_all(fd, &s_net, sizeof(s_net));
    sizes[(size_t)i] = (int64_t)ntoh_u64(s_net);
  }

  uint64_t nbytes_net = 0;
  read_all(fd, &nbytes_net, sizeof(nbytes_net));
  uint64_t nbytes = ntoh_u64(nbytes_net);

  auto dtype = i32_to_scalar_type(dtype_i);
  auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

  torch::Tensor cpu = torch::empty(sizes, opts);
  if ((uint64_t)cpu.nbytes() != nbytes) {
    throw std::runtime_error("recv_tensor: nbytes mismatch");
  }

  read_all(fd, cpu.data_ptr(), (size_t)nbytes);
  return cpu;
}

TcpClient::TcpClient(const std::string& host, int port) {
  struct addrinfo hints;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo* res = nullptr;
  const std::string port_s = std::to_string(port);
  int rc = getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res);
  if (rc != 0 || !res) {
    throw std::runtime_error("getaddrinfo failed");
  }

  fd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd_ < 0) {
    freeaddrinfo(res);
    throw_sys("socket");
  }

  if (::connect(fd_, res->ai_addr, res->ai_addrlen) < 0) {
    freeaddrinfo(res);
    throw_sys("connect");
  }

  freeaddrinfo(res);
}

TcpClient::~TcpClient() {
  if (fd_ >= 0) ::close(fd_);
}

void TcpClient::send_activation(const ActivationPacket& p) {
  // Simple length-prefixed message: write fields directly.
  int32_t version = htonl((uint32_t)p.version);
  int32_t stage_from = htonl((uint32_t)p.stage_from);
  int32_t stage_to = htonl((uint32_t)p.stage_to);

  uint64_t step_net = hton_u64((uint64_t)p.step);
  uint64_t pos_net  = hton_u64((uint64_t)p.pos);

  write_all(fd_, &version, sizeof(version));
  write_all(fd_, &stage_from, sizeof(stage_from));
  write_all(fd_, &stage_to, sizeof(stage_to));
  write_all(fd_, &step_net, sizeof(step_net));
  write_all(fd_, &pos_net, sizeof(pos_net));

  send_tensor(fd_, p.hidden);
  send_tensor(fd_, p.attn_mask.value_or(torch::Tensor()));
}

ActivationPacket TcpClient::recv_activation() {
  ActivationPacket p;

  int32_t version=0, stage_from=0, stage_to=0;
  uint64_t step_net=0, pos_net=0;

  read_all(fd_, &version, sizeof(version));
  read_all(fd_, &stage_from, sizeof(stage_from));
  read_all(fd_, &stage_to, sizeof(stage_to));
  read_all(fd_, &step_net, sizeof(step_net));
  read_all(fd_, &pos_net, sizeof(pos_net));

  p.version = (int32_t)ntohl((uint32_t)version);
  p.stage_from = (int32_t)ntohl((uint32_t)stage_from);
  p.stage_to   = (int32_t)ntohl((uint32_t)stage_to);
  p.step = (int64_t)ntoh_u64(step_net);
  p.pos  = (int64_t)ntoh_u64(pos_net);

  p.hidden = recv_tensor(fd_);
  auto m = recv_tensor(fd_);
  if (m.defined()) p.attn_mask = m;

  return p;
}

TcpServer::TcpServer(int port) {
  fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd_ < 0) throw_sys("socket");

  int opt = 1;
  if (::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    throw_sys("setsockopt");
  }

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)port);

  if (::bind(fd_, (sockaddr*)&addr, sizeof(addr)) < 0) throw_sys("bind");
  if (::listen(fd_, 16) < 0) throw_sys("listen");
}

TcpServer::~TcpServer() {
  if (fd_ >= 0) ::close(fd_);
}

int TcpServer::accept_one() {
  int cfd = ::accept(fd_, nullptr, nullptr);
  if (cfd < 0) throw_sys("accept");
  return cfd;
}

TcpConn::TcpConn(int fd) : fd_(fd) {}
TcpConn::~TcpConn() { if (fd_ >= 0) ::close(fd_); }

void TcpConn::send_activation(const ActivationPacket& p) {
  TcpClient tmp("127.0.0.1", 1);
  (void)tmp;
  // Not used. Use the standalone helpers below.
  throw std::runtime_error("TcpConn::send_activation not implemented");
}

ActivationPacket TcpConn::recv_activation() {
  ActivationPacket p;

  int32_t version=0, stage_from=0, stage_to=0;
  uint64_t step_net=0, pos_net=0;

  read_all(fd_, &version, sizeof(version));
  read_all(fd_, &stage_from, sizeof(stage_from));
  read_all(fd_, &stage_to, sizeof(stage_to));
  read_all(fd_, &step_net, sizeof(step_net));
  read_all(fd_, &pos_net, sizeof(pos_net));

  p.version = (int32_t)ntohl((uint32_t)version);
  p.stage_from = (int32_t)ntohl((uint32_t)stage_from);
  p.stage_to   = (int32_t)ntohl((uint32_t)stage_to);
  p.step = (int64_t)ntoh_u64(step_net);
  p.pos  = (int64_t)ntoh_u64(pos_net);

  p.hidden = recv_tensor(fd_);
  auto m = recv_tensor(fd_);
  if (m.defined()) p.attn_mask = m;

  return p;
}

void TcpConn::send_activation_raw(const ActivationPacket& p) {
  int32_t version = htonl((uint32_t)p.version);
  int32_t stage_from = htonl((uint32_t)p.stage_from);
  int32_t stage_to = htonl((uint32_t)p.stage_to);

  uint64_t step_net = hton_u64((uint64_t)p.step);
  uint64_t pos_net  = hton_u64((uint64_t)p.pos);

  write_all(fd_, &version, sizeof(version));
  write_all(fd_, &stage_from, sizeof(stage_from));
  write_all(fd_, &stage_to, sizeof(stage_to));
  write_all(fd_, &step_net, sizeof(step_net));
  write_all(fd_, &pos_net, sizeof(pos_net));

  send_tensor(fd_, p.hidden);
  send_tensor(fd_, p.attn_mask.value_or(torch::Tensor()));
}

} // namespace qwen
