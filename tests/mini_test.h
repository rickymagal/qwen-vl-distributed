#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TEST_FAIL(...) do { std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); return 1; } while (0)

#define CHECK_TRUE(cond) do { \
  if (!(cond)) { \
    TEST_FAIL("%s:%d: CHECK_TRUE failed: %s", __FILE__, __LINE__, #cond); \
  } \
} while (0)

#define CHECK_EQ(a, b) do { \
  auto _va = (a); \
  auto _vb = (b); \
  if (!(_va == _vb)) { \
    std::fprintf(stderr, "%s:%d: CHECK_EQ failed: %s == %s (got %lld vs %lld)\n", __FILE__, __LINE__, #a, #b, \
      (long long)_va, (long long)_vb); \
    return 1; \
  } \
} while (0)

#define CHECK_NEAR(a, b, eps) do { \
  double _da = (double)(a); \
  double _db = (double)(b); \
  double _de = (double)(eps); \
  if (std::fabs(_da - _db) > _de) { \
    std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: |%s - %s| <= %s (got %.9g vs %.9g, diff=%.9g)\n", __FILE__, __LINE__, \
      #a, #b, #eps, _da, _db, std::fabs(_da - _db)); \
    return 1; \
  } \
} while (0)

#define SKIP_IF(cond, msg) do { \
  if (cond) { \
    std::fprintf(stderr, "%s:%d: SKIP: %s\n", __FILE__, __LINE__, msg); \
    return 0; \
  } \
} while (0)
