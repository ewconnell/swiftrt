//
//  Float16.swift
//  SwiftRTCore
//
//  Created by Edward Connell on 11/27/20.
//

import Foundation

// Float16 is not supported on MacOS right now, so we emulate on MacOS.
// It is supported on other platforms
#if !os(macOS)
extension Float16: StorageElement {
  public typealias Stored = Self
  public typealias Value = Self
}

#else

public struct Float16:
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
    self = Self.float2float16_rn(v)
  }
  
  @inlinable public init() { x = 0 }

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
  @inlinable public static func float2float16_rn(_ v: Float) -> Float16 {
    let x = v.bitPattern
    let u: UInt32 = x & 0x7fffffff
    var remainder, shift, lsb, lsb_s1, lsb_m1:UInt32
    var sign, exponent, mantissa: UInt32
    
    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
      return Float16(bitPattern: UInt16(0x7fff));
    }
    
    sign = ((x >> 16) & UInt32(0x8000))
    
    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
      return Float16(bitPattern: UInt16(sign | UInt32(0x7c00)))
    }
    if (u < 0x33000001) {
      return Float16(bitPattern: UInt16(sign | 0x0000))
    }
    
    exponent = ((u >> 23) & 0xff)
    mantissa = (u & 0x7fffff)
    
    if (exponent > 0x70) {
      shift = 13
      exponent -= 0x70
    } else {
      shift = 0x7e - exponent
      exponent = 0
      mantissa |= 0x800000
    }
    lsb    = (1 << shift)
    lsb_s1 = (lsb >> 1)
    lsb_m1 = (lsb - 1)
    
    // Round to nearest even.
    remainder = (mantissa & lsb_m1)
    mantissa >>= shift
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1) != 0)) {
      mantissa += 1
      if ((mantissa & 0x3ff) == 0) {
        exponent += 1
        mantissa = 0
      }
    }
    
    return Float16(bitPattern: UInt16(sign | (exponent << 10) | mantissa))
  }
  
  //--------------------------------------------------------------------------
  @inlinable public static func float162float(_ v: Float16) -> Float {
    var sign     = UInt32((v.x >> 15) & 1)
    var exponent = UInt32((v.x >> 10) & 0x1f)
    var mantissa = UInt32(v.x & 0x3ff) << 13
    
    if exponent == 0x1f {  /* NaN or Inf */
      if mantissa != 0 {
        sign = 0
        mantissa = UInt32(0x7fffff)
      } else {
        mantissa = 0
      }
      exponent = 0xff
    } else if exponent == 0 {  /* Denorm or Zero */
      if mantissa != 0 {
        var msb: UInt32
        exponent = 0x71
        repeat {
          msb = (mantissa & 0x400000)
          mantissa <<= 1  /* normalize */
          exponent -= 1
        } while msb == 0
        mantissa &= 0x7fffff  /* 1.mantissa is implicit */
      }
    } else {
      exponent += 0x70
    }
    
    return Float(bitPattern: UInt32((sign << 31) | (exponent << 23) | mantissa))
  }
  
  //--------------------------------------------------------------------------
  // functions
  @inlinable public static func < (lhs: Float16, rhs: Float16) -> Bool {
    Float(lhs) < Float(rhs)
  }
  
  @inlinable public static func == (lhs: Float16, rhs: Float16) -> Bool {
    lhs.x == rhs.x
  }
  
  // operators
  @inlinable public static func + (lhs: Float16, rhs: Float16) -> Float16 {
    Float16(Float(lhs) + Float(rhs))
  }
  
  @inlinable public static func - (lhs: Float16, rhs: Float16) -> Float16 {
    Float16(Float(lhs) - Float(rhs))
  }
  
  @inlinable public static func * (lhs: Float16, rhs: Float16) -> Float16 {
    Float16(Float(lhs) * Float(rhs))
  }
  
  @inlinable public static func / (lhs: Float16, rhs: Float16) -> Float16 {
    Float16(Float(lhs) / Float(rhs))
  }
  
}

extension Float {
  @inlinable public init(_ value: Float16) {
    self.init(Float16.float162float(value))
  }
}

//==============================================================================
// Float16
extension Float16: StorageElement {
  public typealias Stored = Self
  public typealias Value = Float
  
  @inlinable public static func storedIndex(_ index: Int) -> Int { index }
  @inlinable public static func storedCount(_ count: Int) -> Int { count }
  @inlinable public static func alignment(_ index: Int) -> Int { 0 }
  
  //-------------------------------------
  // accessors
  @inlinable public static func value(
    at index: Int, from stored: Self
  ) -> Float { Float(stored) }
  
  @inlinable public static func store(
    value: Float, at index: Int, to stored: inout Self
  ) { stored = Self(value) }
  
  @inlinable public static func stored(value: Float) -> Self { Self(value) }
  
  @inlinable public static func storedRange(start: Int, count: Int)
  -> (storedStart: Int, storedCount: Int)
  { (start, count) }
  
  @inlinable public static func getValue(
    from buffer: UnsafeBufferPointer<Float16>,
    at index: Int
  ) -> Float {
    Float(buffer[index])
  }
  
  @inlinable public static func set(
    value: Float,
    in buffer: UnsafeMutableBufferPointer<Float16>,
    at index: Int
  ) {
    buffer[index] = Float16(value)
  }
}
#endif
