#include "core/hf_config.h"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qwen {
namespace {

struct JsonValue;

using JsonObject = std::unordered_map<std::string, JsonValue>;
using JsonArray  = std::vector<JsonValue>;

struct JsonValue {
  enum class Type {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object
  };

  Type type = Type::Null;

  bool b = false;
  double n = 0.0;
  std::string s;
  JsonArray a;
  JsonObject o;

  JsonValue() = default;

  static JsonValue make_null() {
    JsonValue v;
    v.type = Type::Null;
    return v;
  }
  static JsonValue make_bool(bool x) {
    JsonValue v;
    v.type = Type::Bool;
    v.b = x;
    return v;
  }
  static JsonValue make_number(double x) {
    JsonValue v;
    v.type = Type::Number;
    v.n = x;
    return v;
  }
  static JsonValue make_string(std::string x) {
    JsonValue v;
    v.type = Type::String;
    v.s = std::move(x);
    return v;
  }
  static JsonValue make_array(JsonArray x) {
    JsonValue v;
    v.type = Type::Array;
    v.a = std::move(x);
    return v;
  }
  static JsonValue make_object(JsonObject x) {
    JsonValue v;
    v.type = Type::Object;
    v.o = std::move(x);
    return v;
  }
};

class JsonParser {
public:
  explicit JsonParser(std::string src) : src_(std::move(src)) {}

  JsonValue parse_root() {
    idx_ = 0;
    skip_ws();
    JsonValue v = parse_value();
    skip_ws();
    if (idx_ != src_.size()) {
      throw std::runtime_error("hf_config: trailing characters after JSON root");
    }
    return v;
  }

private:
  void skip_ws() {
    while (idx_ < src_.size()) {
      const unsigned char c = static_cast<unsigned char>(src_[idx_]);
      if (!std::isspace(c)) break;
      ++idx_;
    }
  }

  char peek() const {
    if (idx_ >= src_.size()) return '\0';
    return src_[idx_];
  }

  char get() {
    if (idx_ >= src_.size()) {
      throw std::runtime_error("hf_config: unexpected end of input");
    }
    return src_[idx_++];
  }

  void expect(char c) {
    const char got = get();
    if (got != c) {
      std::ostringstream oss;
      oss << "hf_config: expected '" << c << "' but got '" << got << "'";
      throw std::runtime_error(oss.str());
    }
  }

  JsonValue parse_value() {
    skip_ws();
    const char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return JsonValue::make_string(parse_string());
    if (c == '-' || (c >= '0' && c <= '9')) return JsonValue::make_number(parse_number());
    if (starts_with("true"))  { idx_ += 4; return JsonValue::make_bool(true); }
    if (starts_with("false")) { idx_ += 5; return JsonValue::make_bool(false); }
    if (starts_with("null"))  { idx_ += 4; return JsonValue::make_null(); }

    std::ostringstream oss;
    oss << "hf_config: invalid JSON value at offset " << idx_;
    throw std::runtime_error(oss.str());
  }

  bool starts_with(const char* lit) const {
    size_t i = 0;
    while (lit[i] != '\0') {
      if (idx_ + i >= src_.size()) return false;
      if (src_[idx_ + i] != lit[i]) return false;
      ++i;
    }
    return true;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    out.reserve(32);

    while (true) {
      if (idx_ >= src_.size()) {
        throw std::runtime_error("hf_config: unterminated string");
      }
      char c = get();
      if (c == '"') break;

      if (c == '\\') {
        if (idx_ >= src_.size()) throw std::runtime_error("hf_config: bad escape");
        char e = get();
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            // Minimal \uXXXX support: we parse the code unit and emit UTF-8 for BMP.
            // For config files this is typically not needed, but we keep it correct enough.
            uint32_t code = 0;
            for (int i = 0; i < 4; ++i) {
              if (idx_ >= src_.size()) throw std::runtime_error("hf_config: bad unicode escape");
              const char h = get();
              code <<= 4;
              if (h >= '0' && h <= '9') code |= static_cast<uint32_t>(h - '0');
              else if (h >= 'a' && h <= 'f') code |= static_cast<uint32_t>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') code |= static_cast<uint32_t>(h - 'A' + 10);
              else throw std::runtime_error("hf_config: bad unicode escape");
            }
            append_utf8(out, code);
            break;
          }
          default:
            throw std::runtime_error("hf_config: unsupported escape sequence");
        }
      } else {
        out.push_back(c);
      }
    }
    return out;
  }

  static void append_utf8(std::string& out, uint32_t code) {
    if (code <= 0x7F) {
      out.push_back(static_cast<char>(code));
    } else if (code <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code <= 0xFFFF) {
      out.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | ((code >> 18) & 0x07)));
      out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  double parse_number() {
    // JSON number: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?
    const size_t start = idx_;
    if (peek() == '-') ++idx_;

    if (idx_ >= src_.size()) throw std::runtime_error("hf_config: bad number");
    if (src_[idx_] == '0') {
      ++idx_;
    } else if (src_[idx_] >= '1' && src_[idx_] <= '9') {
      while (idx_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[idx_]))) ++idx_;
    } else {
      throw std::runtime_error("hf_config: bad number");
    }

    if (idx_ < src_.size() && src_[idx_] == '.') {
      ++idx_;
      if (idx_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[idx_]))) {
        throw std::runtime_error("hf_config: bad number fraction");
      }
      while (idx_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[idx_]))) ++idx_;
    }

    if (idx_ < src_.size() && (src_[idx_] == 'e' || src_[idx_] == 'E')) {
      ++idx_;
      if (idx_ < src_.size() && (src_[idx_] == '+' || src_[idx_] == '-')) ++idx_;
      if (idx_ >= src_.size() || !std::isdigit(static_cast<unsigned char>(src_[idx_]))) {
        throw std::runtime_error("hf_config: bad number exponent");
      }
      while (idx_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[idx_]))) ++idx_;
    }

    const std::string num_str = src_.substr(start, idx_ - start);
    char* endp = nullptr;
    const double v = std::strtod(num_str.c_str(), &endp);
    if (!endp || *endp != '\0') {
      throw std::runtime_error("hf_config: failed to parse number");
    }
    return v;
  }

  JsonValue parse_array() {
    expect('[');
    skip_ws();
    JsonArray arr;
    if (peek() == ']') {
      get();
      return JsonValue::make_array(std::move(arr));
    }

    while (true) {
      skip_ws();
      arr.push_back(parse_value());
      skip_ws();
      const char c = get();
      if (c == ']') break;
      if (c != ',') throw std::runtime_error("hf_config: expected ',' or ']' in array");
    }

    return JsonValue::make_array(std::move(arr));
  }

  JsonValue parse_object() {
    expect('{');
    skip_ws();
    JsonObject obj;
    if (peek() == '}') {
      get();
      return JsonValue::make_object(std::move(obj));
    }

    while (true) {
      skip_ws();
      if (peek() != '"') throw std::runtime_error("hf_config: expected string key in object");
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      JsonValue val = parse_value();
      obj.emplace(std::move(key), std::move(val));
      skip_ws();
      const char c = get();
      if (c == '}') break;
      if (c != ',') throw std::runtime_error("hf_config: expected ',' or '}' in object");
    }

    return JsonValue::make_object(std::move(obj));
  }

  std::string src_;
  size_t idx_ = 0;
};

static std::string read_text_file(const std::string& path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    throw std::runtime_error("hf_config: failed to open file: " + path);
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static const JsonValue* obj_get(const JsonObject& o, const std::string& k) {
  auto it = o.find(k);
  if (it == o.end()) return nullptr;
  return &it->second;
}

static const JsonObject* as_object_ptr(const JsonValue& v) {
  if (v.type != JsonValue::Type::Object) return nullptr;
  return &v.o;
}

static const JsonArray* as_array_ptr(const JsonValue& v) {
  if (v.type != JsonValue::Type::Array) return nullptr;
  return &v.a;
}

static bool as_bool(const JsonValue& v, bool* out) {
  if (v.type == JsonValue::Type::Bool) {
    *out = v.b;
    return true;
  }
  return false;
}

static bool as_i64(const JsonValue& v, int64_t* out) {
  if (v.type == JsonValue::Type::Number) {
    const double x = v.n;
    if (x < static_cast<double>(std::numeric_limits<int64_t>::min())) return false;
    if (x > static_cast<double>(std::numeric_limits<int64_t>::max())) return false;
    *out = static_cast<int64_t>(x);
    return true;
  }
  return false;
}

static bool as_i32(const JsonValue& v, int32_t* out) {
  int64_t tmp = 0;
  if (!as_i64(v, &tmp)) return false;
  if (tmp < static_cast<int64_t>(std::numeric_limits<int32_t>::min())) return false;
  if (tmp > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) return false;
  *out = static_cast<int32_t>(tmp);
  return true;
}

static bool as_f32(const JsonValue& v, float* out) {
  if (v.type == JsonValue::Type::Number) {
    const double x = v.n;
    if (x < -static_cast<double>(std::numeric_limits<float>::max())) return false;
    if (x >  static_cast<double>(std::numeric_limits<float>::max())) return false;
    *out = static_cast<float>(x);
    return true;
  }
  return false;
}

static bool as_string(const JsonValue& v, std::string* out) {
  if (v.type == JsonValue::Type::String) {
    *out = v.s;
    return true;
  }
  return false;
}

static void parse_vision_config(const JsonObject& root, ModelConfig* cfg) {
  const JsonValue* vcfg_val = obj_get(root, "vision_config");
  if (!vcfg_val) return;
  const JsonObject* vcfg = as_object_ptr(*vcfg_val);
  if (!vcfg) return;

  {
    const JsonValue* hv = obj_get(*vcfg, "hidden_size");
    if (hv) (void)as_i32(*hv, &cfg->vision_hidden_size);
  }
  {
    const JsonValue* nl = obj_get(*vcfg, "num_hidden_layers");
    if (nl) (void)as_i32(*nl, &cfg->vision_num_layers);
  }

  // Some configs store these fields with alternate keys.
  if (cfg->vision_hidden_size <= 0) {
    const JsonValue* hv2 = obj_get(*vcfg, "vision_hidden_size");
    if (hv2) (void)as_i32(*hv2, &cfg->vision_hidden_size);
  }
  if (cfg->vision_num_layers <= 0) {
    const JsonValue* nl2 = obj_get(*vcfg, "vision_num_layers");
    if (nl2) (void)as_i32(*nl2, &cfg->vision_num_layers);
  }
}

static void parse_moe_fields(const JsonObject& root, ModelConfig* cfg) {
  // Common HF keys seen across Qwen-style configs.
  // We keep this tolerant: if any "num_experts" style field exists and > 0, enable MoE.
  const char* expert_keys[] = {
    "num_experts",
    "moe_num_experts",
    "num_local_experts",
    "n_experts"
  };
  for (const char* k : expert_keys) {
    const JsonValue* v = obj_get(root, k);
    if (v) {
      int32_t x = 0;
      if (as_i32(*v, &x) && x > 0) {
        cfg->num_experts = x;
        break;
      }
    }
  }

  const char* topk_keys[] = {
    "num_experts_per_tok",
    "top_k",
    "moe_top_k",
    "router_top_k"
  };
  for (const char* k : topk_keys) {
    const JsonValue* v = obj_get(root, k);
    if (v) {
      int32_t x = 0;
      if (as_i32(*v, &x) && x > 0) {
        cfg->top_k = x;
        break;
      }
    }
  }

  // Some configs contain MoE params nested under a "moe" object.
  const JsonValue* moe_val = obj_get(root, "moe");
  if (moe_val) {
    const JsonObject* moe_obj = as_object_ptr(*moe_val);
    if (moe_obj) {
      if (cfg->num_experts <= 0) {
        const JsonValue* v = obj_get(*moe_obj, "num_experts");
        if (v) (void)as_i32(*v, &cfg->num_experts);
      }
      if (cfg->top_k <= 0) {
        const JsonValue* v = obj_get(*moe_obj, "top_k");
        if (v) (void)as_i32(*v, &cfg->top_k);
      }
    }
  }

  cfg->use_moe = (cfg->num_experts > 0 && cfg->top_k > 0);
}

static void apply_root_fields(const JsonObject& root, ModelConfig* cfg) {
  // Identity / dtype
  {
    const JsonValue* v = obj_get(root, "name_or_path");
    if (v) (void)as_string(*v, &cfg->model_id);
  }
  if (cfg->model_id.empty()) {
    const JsonValue* v = obj_get(root, "model_type");
    if (v) (void)as_string(*v, &cfg->model_id);
  }
  {
    const JsonValue* v = obj_get(root, "torch_dtype");
    if (v) (void)as_string(*v, &cfg->dtype);
  }

  // Core text model params
  {
    const JsonValue* v = obj_get(root, "vocab_size");
    if (v) (void)as_i32(*v, &cfg->vocab_size);
  }
  {
    const JsonValue* v = obj_get(root, "hidden_size");
    if (v) (void)as_i32(*v, &cfg->hidden_size);
  }
  {
    const JsonValue* v = obj_get(root, "num_hidden_layers");
    if (v) (void)as_i32(*v, &cfg->num_hidden_layers);
  }
  {
    const JsonValue* v = obj_get(root, "num_attention_heads");
    if (v) (void)as_i32(*v, &cfg->num_attention_heads);
  }
  {
    const JsonValue* v = obj_get(root, "num_key_value_heads");
    if (v) (void)as_i32(*v, &cfg->num_key_value_heads);
  }
  {
    const JsonValue* v = obj_get(root, "intermediate_size");
    if (v) (void)as_i32(*v, &cfg->intermediate_size);
  }

  // Sequence length
  {
    const JsonValue* v = obj_get(root, "max_position_embeddings");
    if (v) (void)as_i32(*v, &cfg->max_seq_len);
  }
  if (cfg->max_seq_len <= 0) {
    const JsonValue* v = obj_get(root, "seq_length");
    if (v) (void)as_i32(*v, &cfg->max_seq_len);
  }
  if (cfg->max_seq_len <= 0) {
    const JsonValue* v = obj_get(root, "max_sequence_length");
    if (v) (void)as_i32(*v, &cfg->max_seq_len);
  }

  // RoPE params
  {
    const JsonValue* v = obj_get(root, "rope_theta");
    if (v) (void)as_f32(*v, &cfg->rope_theta);
  }
  {
    const JsonValue* v = obj_get(root, "rotary_emb_base");
    if (v && cfg->rope_theta == 0.0f) (void)as_f32(*v, &cfg->rope_theta);
  }
  {
    const JsonValue* v = obj_get(root, "rope_dim");
    if (v) (void)as_i32(*v, &cfg->rope_dim);
  }

  // Some configs place rope scaling under "rope_scaling".
  const JsonValue* rs = obj_get(root, "rope_scaling");
  if (rs) {
    const JsonObject* rso = as_object_ptr(*rs);
    if (rso) {
      const JsonValue* v = obj_get(*rso, "rope_theta");
      if (v) (void)as_f32(*v, &cfg->rope_theta);
      // Do not attempt to derive rope_dim from scaling; model code can compute defaults.
    }
  }

  // Capacity
  {
    const JsonValue* v = obj_get(root, "max_batch_size");
    if (v) (void)as_i32(*v, &cfg->max_batch);
  }

  // MoE + vision
  parse_moe_fields(root, cfg);
  parse_vision_config(root, cfg);

  // If num_key_value_heads is missing, default to num_attention_heads.
  if (cfg->num_key_value_heads <= 0 && cfg->num_attention_heads > 0) {
    cfg->num_key_value_heads = cfg->num_attention_heads;
  }
}

static ModelConfig parse_model_config_from_json_text(const std::string& text) {
  JsonParser p(text);
  JsonValue root_val = p.parse_root();
  if (root_val.type != JsonValue::Type::Object) {
    throw std::runtime_error("hf_config: JSON root must be an object");
  }

  ModelConfig cfg; // uses defaults from core/config.h
  apply_root_fields(root_val.o, &cfg);

  // Basic sanity checks (do not over-constrain; exporters may omit fields)
  if (cfg.hidden_size <= 0) {
    throw std::runtime_error("hf_config: missing or invalid hidden_size");
  }
  if (cfg.num_attention_heads <= 0) {
    throw std::runtime_error("hf_config: missing or invalid num_attention_heads");
  }
  if (cfg.vocab_size <= 0) {
    throw std::runtime_error("hf_config: missing or invalid vocab_size");
  }
  if (cfg.num_hidden_layers <= 0) {
    throw std::runtime_error("hf_config: missing or invalid num_hidden_layers");
  }
  return cfg;
}

} // namespace

bool try_load_hf_config_json(const std::string& path, ModelConfig* out, std::string* err) {
  if (!out) {
    if (err) *err = "hf_config: output pointer is null";
    return false;
  }

  try {
    const std::string text = read_text_file(path);
    *out = parse_model_config_from_json_text(text);
    return true;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }
}

ModelConfig load_hf_config_json(const std::string& path) {
  ModelConfig cfg;
  std::string err;
  if (!try_load_hf_config_json(path, &cfg, &err)) {
    throw std::runtime_error(err.empty() ? "hf_config: failed to load" : err);
  }
  return cfg;
}

} // namespace qwen
