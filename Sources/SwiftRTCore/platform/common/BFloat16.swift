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
// inspired by the C implementation of Paulius Micikevicius pauliusm@nvidia.com
//

// *** TODO *** we need to decide how numerically correct we want to
// emulate Float16 and BFloat16

public struct BFloat16:
  Equatable,
  Comparable,
  AdditiveArithmetic,
  ExpressibleByFloatLiteral,
  ExpressibleByIntegerLiteral
{
  // properties
  public let x: UInt16

  //--------------------------------------------------------------------------
  // initializers
  @inlinable public init(_ v: Float) {
    self = Self.float2bfloat16_rn(v)
  }

  @inlinable public init(bitPattern: UInt16) { x = bitPattern }
  @inlinable public init?(_ string: String) {
    guard let v = Float(string) else { return nil }
    self.init(v)
  }

  @inlinable public init(floatLiteral value: Double) {
    self.init(Float(value))
  }

  @inlinable public init(integerLiteral value: Int) {
    self.init(Float(value))
  }

  //--------------------------------------------------------------------------
  @inlinable public static func float2bfloat16_rn(_ v: Float) -> BFloat16 {
    var raw: UInt16 = 0
    var remainder: UInt32 = 0
    raw = Self.float2bfloat16(v, &remainder)

    if (remainder > UInt32(0x8000_0000))
      || ((remainder == UInt32(0x8000_0000)) && ((raw & UInt16(1)) != UInt32(0)))
    {
      raw += 1
    }
    return BFloat16(bitPattern: raw)
  }

  //--------------------------------------------------------------------------
  @usableFromInline static func float2bfloat16(
    _ f: Float,
    _ rem: inout UInt32
  ) -> UInt16 {
    let x = UInt32(f.bitPattern)

    if (x & UInt32(0x7fff_ffff)) > UInt32(0x7f80_0000) {
      rem = 0
      return UInt16(0x7fff)
    } else {
      rem = x << 16
      return UInt16(x >> 16)
    }
  }

  //--------------------------------------------------------------------------
  @inlinable public static func bfloat162float(_ v: BFloat16) -> Float {
    Float(bitPattern: UInt32(v.x) << 16)
  }

  //--------------------------------------------------------------------------
  // functions
  @inlinable public static func < (lhs: BFloat16, rhs: BFloat16) -> Bool {
    Float(lhs) < Float(rhs)
  }

  @inlinable public static func == (lhs: BFloat16, rhs: BFloat16) -> Bool {
    lhs.x == rhs.x
  }

  // operators
  @inlinable public static func + (lhs: BFloat16, rhs: BFloat16) -> BFloat16 {
    BFloat16(Float(lhs) + Float(rhs))
  }

  @inlinable public static func - (lhs: BFloat16, rhs: BFloat16) -> BFloat16 {
    BFloat16(Float(lhs) - Float(rhs))
  }

  @inlinable public static func * (lhs: BFloat16, rhs: BFloat16) -> BFloat16 {
    BFloat16(Float(lhs) * Float(rhs))
  }

  @inlinable public static func / (lhs: BFloat16, rhs: BFloat16) -> BFloat16 {
    BFloat16(Float(lhs) / Float(rhs))
  }

}

extension Float {
  @inlinable public init(_ value: BFloat16) {
    self.init(BFloat16.bfloat162float(value))
  }
}
