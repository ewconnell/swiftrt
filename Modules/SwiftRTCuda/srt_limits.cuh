//******************************************************************************
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#pragma once
#include <limits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "float16.cuh"
#include "bfloat16.cuh"

//==============================================================================
// float16 limits
//==============================================================================

namespace std {

template <> struct numeric_limits<half> {
  static constexpr bool is_specialized = true;

    __HOSTDEVICE_INLINE__ static half min() noexcept {
        return __FLT_MIN__;
    }

    __HOSTDEVICE_INLINE__ static half max() noexcept {
        return __FLT_MAX__;
    }

// #if __cplusplus >= 201103L
//   static constexpr float lowest() noexcept { return -__FLT_MAX__; }
// #endif

//   static constexpr int digits = __FLT_MANT_DIG__;
//   static constexpr int digits10 = __FLT_DIG__;
// #if __cplusplus >= 201103L
//   static constexpr int max_digits10 = __glibcxx_max_digits10(__FLT_MANT_DIG__);
// #endif
//   static constexpr bool is_signed = true;
//   static constexpr bool is_integer = false;
//   static constexpr bool is_exact = false;
//   static constexpr int radix = __FLT_RADIX__;

//   static constexpr float epsilon() _GLIBCXX_USE_NOEXCEPT {
//     return __FLT_EPSILON__;
//   }

//   static constexpr float round_error() _GLIBCXX_USE_NOEXCEPT {
//     return 0.5F;
//   }

//   static constexpr int min_exponent = __FLT_MIN_EXP__;
//   static constexpr int min_exponent10 = __FLT_MIN_10_EXP__;
//   static constexpr int max_exponent = __FLT_MAX_EXP__;
//   static constexpr int max_exponent10 = __FLT_MAX_10_EXP__;

//   static constexpr bool has_infinity = __FLT_HAS_INFINITY__;
//   static constexpr bool has_quiet_NaN = __FLT_HAS_QUIET_NAN__;
//   static constexpr bool has_signaling_NaN = has_quiet_NaN;
//   static constexpr float_denorm_style has_denorm =
//       bool(__FLT_HAS_DENORM__) ? denorm_present : denorm_absent;
//   static constexpr bool has_denorm_loss =
//       __glibcxx_float_has_denorm_loss;

    __HOSTDEVICE_INLINE__ static float16 infinity() noexcept {
        return __builtin_huge_valf();
    }

//   static constexpr float quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
//     return __builtin_nanf("");
//   }

//   static constexpr float signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
//     return __builtin_nansf("");
//   }

//   static constexpr float denorm_min() _GLIBCXX_USE_NOEXCEPT {
//     return __FLT_DENORM_MIN__;
//   }

//   static constexpr bool is_iec559 =
//       has_infinity && has_quiet_NaN && has_denorm == denorm_present;
//   static constexpr bool is_bounded = true;
//   static constexpr bool is_modulo = false;

//   static constexpr bool traps = __glibcxx_float_traps;
//   static constexpr bool tinyness_before =
//       __glibcxx_float_tinyness_before;
//   static constexpr float_round_style round_style =
//       round_to_nearest;
};
} // std

//==============================================================================
// bfloat16 limits
//==============================================================================

typedef __nv_bfloat16 bfloat16;

namespace std {

template <> struct numeric_limits<bfloat16> {
  static constexpr bool is_specialized = true;

    __HOSTDEVICE_INLINE__ static bfloat16 min() noexcept {
        return __FLT_MIN__;
    }

    __HOSTDEVICE_INLINE__ static bfloat16 max() noexcept {
        return __FLT_MAX__;
    }

// #if __cplusplus >= 201103L
//   static constexpr float lowest() noexcept { return -__FLT_MAX__; }
// #endif

//   static constexpr int digits = __FLT_MANT_DIG__;
//   static constexpr int digits10 = __FLT_DIG__;
// #if __cplusplus >= 201103L
//   static constexpr int max_digits10 = __glibcxx_max_digits10(__FLT_MANT_DIG__);
// #endif
//   static constexpr bool is_signed = true;
//   static constexpr bool is_integer = false;
//   static constexpr bool is_exact = false;
//   static constexpr int radix = __FLT_RADIX__;

//   static constexpr float epsilon() _GLIBCXX_USE_NOEXCEPT {
//     return __FLT_EPSILON__;
//   }

//   static constexpr float round_error() _GLIBCXX_USE_NOEXCEPT {
//     return 0.5F;
//   }

//   static constexpr int min_exponent = __FLT_MIN_EXP__;
//   static constexpr int min_exponent10 = __FLT_MIN_10_EXP__;
//   static constexpr int max_exponent = __FLT_MAX_EXP__;
//   static constexpr int max_exponent10 = __FLT_MAX_10_EXP__;

//   static constexpr bool has_infinity = __FLT_HAS_INFINITY__;
//   static constexpr bool has_quiet_NaN = __FLT_HAS_QUIET_NAN__;
//   static constexpr bool has_signaling_NaN = has_quiet_NaN;
//   static constexpr float_denorm_style has_denorm =
//       bool(__FLT_HAS_DENORM__) ? denorm_present : denorm_absent;
//   static constexpr bool has_denorm_loss =
//       __glibcxx_float_has_denorm_loss;

    __HOSTDEVICE_INLINE__ static bfloat16 infinity() noexcept {
        return __builtin_huge_valf();
    }

//   static constexpr float quiet_NaN() _GLIBCXX_USE_NOEXCEPT {
//     return __builtin_nanf("");
//   }

//   static constexpr float signaling_NaN() _GLIBCXX_USE_NOEXCEPT {
//     return __builtin_nansf("");
//   }

//   static constexpr float denorm_min() _GLIBCXX_USE_NOEXCEPT {
//     return __FLT_DENORM_MIN__;
//   }

//   static constexpr bool is_iec559 =
//       has_infinity && has_quiet_NaN && has_denorm == denorm_present;
//   static constexpr bool is_bounded = true;
//   static constexpr bool is_modulo = false;

//   static constexpr bool traps = __glibcxx_float_traps;
//   static constexpr bool tinyness_before =
//       __glibcxx_float_tinyness_before;
//   static constexpr float_round_style round_style =
//       round_to_nearest;
};
} // std
