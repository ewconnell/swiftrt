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
#include <stdint.h>
#include <vector_types.h>
#include <type_traits>

#include "cuda_macros.cuh"
#include "float16.cuh"
#include "bfloat16.cuh"
#include "complex.cuh"
#include "simd_types.cuh"

//==============================================================================
// extend standard traits
namespace std {
    //--------------------------------------------------------------------------
    // is_floating_point
    template<>
    struct __is_floating_point_helper<float16>: public true_type { };

    template<>
    struct __is_floating_point_helper<float162>: public true_type { };

    template<>
    struct __is_floating_point_helper<bfloat16>: public true_type { };

    template<>
    struct __is_floating_point_helper<bfloat162>: public true_type { };

    //--------------------------------------------------------------------------
    // is_signed
    template<>
    struct __is_signed_helper<float16>: public true_type { };

    template<>
    struct __is_signed_helper<float162>: public true_type { };

    template<>
    struct __is_signed_helper<bfloat16>: public true_type { };

    template<>
    struct __is_signed_helper<bfloat162>: public true_type { };

    //--------------------------------------------------------------------------
    // is_signed
  template<>
    struct __is_integral_helper<char4>
    : public true_type { };

  template<>
    struct __is_integral_helper<uchar4>
    : public true_type { };

  template<>
    struct __is_integral_helper<short2>
    : public true_type { };

  template<>
    struct __is_integral_helper<ushort2>
    : public true_type { };
}


//==============================================================================
// conformance helpers
template<typename A, typename O>
inline constexpr bool isSame() {
    return std::is_same<A,O>::value;
}

template<typename A>
inline constexpr bool isInteger() {
    return std::is_integral<A>::value;
}

template<typename A>
inline constexpr bool isFloating() {
    return std::is_floating_point<A>::value;
}

template<typename A>
inline constexpr bool isComplex() {
    return std::is_same<A, Complex<float>>::value ||
        std::is_same<A, Complex<float16>>::value ||
        std::is_same<A, Complex<bfloat16>>::value;
}

template<typename ComplexType, typename RealType>
inline constexpr bool isComplexRealType() {
    if constexpr (isComplex<ComplexType>()) {
        return std::is_same<typename ComplexType::RealType, RealType>::value;
    }
    return false;
}

template<typename A>
inline constexpr bool isBool() {
    return std::is_same<A,bool>::value ||
        std::is_same<A,bool2>::value || std::is_same<A,bool4>::value;
}

template<typename A>
inline constexpr bool isNumeric() {
    return isInteger<A>() || isFloating<A>() || isComplex<A>();
}

template<typename A>
inline constexpr bool isComparable() {
    return isNumeric<A>();
}

template<typename T>
inline constexpr bool isEquatable() {
    return isNumeric<T>() || isBool<T>();
}

template<typename A>
inline constexpr bool isSigned() {
    return std::is_signed<A>::value || isFloating<A>() || isComplex<A>();
}

template<typename A>
inline constexpr bool isSignedNumeric() {
    return isSigned<A>() && isNumeric<A>();
}

template<typename A>
inline constexpr bool isPacked() {
    return 
    std::is_same<A,bool2>::value  || std::is_same<A,bool4>::value ||
    std::is_same<A,char4>::value  || std::is_same<A,uchar4>::value ||
    std::is_same<A,short2>::value || std::is_same<A,ushort2>::value ||
    std::is_same<A,half2>::value || std::is_same<A,bfloat162>::value;
}

//==============================================================================
// given an input type A and an output type O, if the input is
// packed, then the corresponding packed respresention of O is defined
template<typename T> struct packed {
    typedef T type;
    inline static T value(const T v) { return v; }
};

template<> struct packed<int8_t> {
    typedef char4 type;
    inline static type value(const int8_t v) {
        type p; p.x = v; p.y = v; p.z = v; p.w = v; return p;
    }
};

template<> struct packed<uint8_t> {
    typedef uchar4 type;
    inline static type value(const uint8_t v) {
        type p; p.x = v; p.y = v; p.z = v; p.w = v; return p;
    }
};

template<> struct packed<int16_t> {
    typedef short2 type;
    inline static type value(const int16_t v) {
        type p; p.x = v; p.y = v; return p;
    }
};

template<> struct packed<uint16_t> {
    typedef ushort2 type;
    inline static type value(const uint16_t v) {
        type p; p.x = v; p.y = v; return p;
    }
};

template<> struct packed<half> {
    typedef half2 type;
    inline static type value(const half v) {
        type p; p.x = v; p.y = v; return p;
    }
};

template<> struct packed<bfloat16> {
    typedef bfloat162 type;
    inline static type value(const bfloat16 v) {
        type p; p.x = v; p.y = v; return p;
    }
};

//--------------------------------------
// given an input type A and an output type O, if the input is
// packed, then the corresponding packed respresention of O is defined
template<typename A, typename O>
struct matching_packed { typedef O type; };
template<> struct matching_packed<char4, bool> { typedef bool4 type; };
template<> struct matching_packed<char4, int8_t> { typedef char4 type; };

template<> struct matching_packed<uchar4, bool> { typedef bool4 type; };
template<> struct matching_packed<uchar4, uint8_t> { typedef uchar4 type; };

template<> struct matching_packed<short2, bool> { typedef bool2 type; };
template<> struct matching_packed<short2, int16_t> { typedef short2 type; };

template<> struct matching_packed<ushort2, bool> { typedef bool2 type; };
template<> struct matching_packed<ushort2, uint16_t> { typedef ushort2 type; };

template<> struct matching_packed<half2, bool> { typedef bool2 type; };
template<> struct matching_packed<half2, half> { typedef half2 type; };

template<> struct matching_packed<bfloat162, bool> { typedef bool2 type; };
template<> struct matching_packed<bfloat162, bfloat16> { typedef bfloat162 type; };

//--------------------------------------
// given an input type A and an output type O, if the input is
// packed, then the corresponding packed respresention of O is defined
template<typename A> struct packing { static const int count = 1; };

template<> struct packing<char4> { static const int count = 4; };
template<> struct packing<const char4> { static const int count = 4; };
template<> struct packing<uchar4> { static const int count = 4; };
template<> struct packing<const uchar4> { static const int count = 4; };
template<> struct packing<bool4> { static const int count = 4; };
template<> struct packing<const bool4> { static const int count = 4; };

template<> struct packing<bool2> { static const int count = 2; };
template<> struct packing<const bool2> { static const int count = 2; };
template<> struct packing<short2> { static const int count = 2; };
template<> struct packing<const short2> { static const int count = 2; };
template<> struct packing<ushort2> { static const int count = 2; };
template<> struct packing<const ushort2> { static const int count = 2; };
template<> struct packing<half2> { static const int count = 2; };
template<> struct packing<const half2> { static const int count = 2; };
template<> struct packing<bfloat162> { static const int count = 2; };
template<> struct packing<const bfloat162> { static const int count = 2; };

// template<typename A, typename O>
// inline constexpr bool canPack() {
//     return packing<A>::count == packing<O>::count;
// }


//==============================================================================
/// init
/// fill all lanes
template<typename T>
__HOSTDEVICE_INLINE__ T init(float v) {
    T t;
    if constexpr (packing<T>::count == 1) {
        t = T(v);
    } else if constexpr (packing<T>::count == 2) {
        t.x = v;
        t.y = v;
    } else if constexpr (packing<T>::count == 4) {
        t.x = v;
        t.y = v;
        t.z = v;
        t.w = v;
    }
    return t;
}

