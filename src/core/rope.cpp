#include "core/rope.h"
#include "core/tensor_utils.h"

namespace qwen {

static torch::Tensor build_inv_freq(int64_t rope_dim, double theta, int device_index) {
  require(rope_dim > 0, "rope_dim must be > 0");
  require((rope_dim % 2) == 0, "rope_dim must be even");

  const int64_t half = rope_dim / 2;

  auto opts = torch::TensorOptions()
                  .dtype(torch::kFloat32)
                  .device(torch::kCUDA, device_index);

  // inv_freq[i] = 1 / (theta^(2i/rope_dim))
  auto i = torch::arange(0, half, opts);
  auto exponent = (2.0f * i) / static_cast<float>(rope_dim);
  auto inv_freq = torch::pow(torch::tensor(theta, opts), -exponent);
  return inv_freq; // [half]
}

RopeTables precompute_cos_sin(int64_t seq_len,
                              int64_t rope_dim,
                              double theta,
                              c10::ScalarType dtype,
                              int device_index) {
  require(seq_len > 0, "seq_len must be > 0");
  require(rope_dim > 0, "rope_dim must be > 0");
  require((rope_dim % 2) == 0, "rope_dim must be even");

  auto inv_freq = build_inv_freq(rope_dim, theta, device_index); // [half]
  auto t_opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCUDA, device_index);

  auto t = torch::arange(0, seq_len, t_opts);      // [T]
  auto freqs = torch::einsum("t,f->tf", {t, inv_freq}); // [T, half]

  auto cos_half = torch::cos(freqs);
  auto sin_half = torch::sin(freqs);

  // Expand to [T, rope_dim] by interleaving
  auto cos = torch::empty({seq_len, rope_dim}, t_opts);
  auto sin = torch::empty({seq_len, rope_dim}, t_opts);

  cos.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, rope_dim, 2)}, cos_half);
  cos.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, rope_dim, 2)}, cos_half);

  sin.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, rope_dim, 2)}, sin_half);
  sin.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, rope_dim, 2)}, sin_half);

  RopeTables out;
  out.rope_dim = rope_dim;
  out.cos = cos.to(dtype);
  out.sin = sin.to(dtype);
  return out;
}

static void rope_rotate_inplace(torch::Tensor x,
                                const torch::Tensor& cos_t,
                                const torch::Tensor& sin_t,
                                int64_t rope_dim) {
  // x: [B, H, T, D]
  // cos_t/sin_t: [T, rope_dim]
  require(x.dim() == 4, "apply_rope expects x dim == 4");
  require(cos_t.dim() == 2 && sin_t.dim() == 2, "cos/sin must be [T, rope_dim]");
  require(cos_t.size(1) == rope_dim && sin_t.size(1) == rope_dim, "rope_dim mismatch");
  require(x.size(2) == cos_t.size(0), "T mismatch between x and cos/sin");
  require((rope_dim % 2) == 0, "rope_dim must be even");
  require(x.size(3) >= rope_dim, "head_dim must be >= rope_dim");

  // Slice first rope_dim
  auto x_rope = x.index({torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice(0, rope_dim)}); // [B,H,T,rope_dim]

  // Reshape to pairs: [B,H,T,rope_dim/2,2]
  auto x_pair = x_rope.view({x_rope.size(0), x_rope.size(1), x_rope.size(2), rope_dim / 2, 2});

  auto x1 = x_pair.select(-1, 0); // [B,H,T,half]
  auto x2 = x_pair.select(-1, 1); // [B,H,T,half]

  // cos/sin half are the even positions
  auto cos_half = cos_t.index({torch::indexing::Slice(), torch::indexing::Slice(0, rope_dim, 2)}); // [T,half]
  auto sin_half = sin_t.index({torch::indexing::Slice(), torch::indexing::Slice(0, rope_dim, 2)}); // [T,half]

  // Broadcast to [1,1,T,half]
  cos_half = cos_half.unsqueeze(0).unsqueeze(0);
  sin_half = sin_half.unsqueeze(0).unsqueeze(0);

  // Rotate
  // y1 = x1*cos - x2*sin
  // y2 = x1*sin + x2*cos
  auto y1 = x1 * cos_half - x2 * sin_half;
  auto y2 = x1 * sin_half + x2 * cos_half;

  // Write back
  x_pair.select(-1, 0).copy_(y1);
  x_pair.select(-1, 1).copy_(y2);
}

void apply_rope_inplace(torch::Tensor q,
                        torch::Tensor k,
                        const RopeTables& tables,
                        int64_t start_pos) {
  require(q.defined() && k.defined(), "q/k must be defined");
  require(q.is_cuda() && k.is_cuda(), "q/k must be CUDA tensors");
  require(q.scalar_type() == tables.cos.scalar_type(), "q dtype must match rope tables dtype");
  require(k.scalar_type() == tables.cos.scalar_type(), "k dtype must match rope tables dtype");
  require(q.dim() == 4 && k.dim() == 4, "q/k must be [B,H,T,D]");
  require(q.size(2) == k.size(2), "q/k must have same T");
  require(q.size(3) == k.size(3), "q/k must have same D");

  const int64_t T = q.size(2);
  const int64_t rope_dim = tables.rope_dim;
  require(start_pos >= 0, "start_pos must be >= 0");
  require(start_pos + T <= tables.cos.size(0), "rope tables too small for requested positions");

  auto cos_t = tables.cos.index({torch::indexing::Slice(start_pos, start_pos + T),
                                 torch::indexing::Slice()}); // [T, rope_dim]
  auto sin_t = tables.sin.index({torch::indexing::Slice(start_pos, start_pos + T),
                                 torch::indexing::Slice()}); // [T, rope_dim]

  rope_rotate_inplace(q, cos_t, sin_t, rope_dim);
  rope_rotate_inplace(k, cos_t, sin_t, rope_dim);
}

} // namespace qwen
