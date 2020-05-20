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
import Foundation

public struct Float16 : Equatable, Comparable, AdditiveArithmetic {
    // properties
    public let x: UInt16

    // 10:5:1
    public static let mantissaMask: UInt16 = 0b0000001111111111
    public static let exponentMask: UInt16 = 0b0111110000000000
    public static let signMask:     UInt16 = 0b1000000000000000
    
    //--------------------------------------------------------------------------
	// initializers
	@inlinable public init() { x = UInt16(0) }
	@inlinable public init(bitPattern: UInt16) { x = bitPattern }
	@inlinable public init?(_ string: String) {
        guard let v = Float(string) else { return nil }
        self = FloatToFloat16Rnd(v)
	}

	@inlinable public init(_ v: UInt8)  { self = FloatToFloat16Rnd(Float(v)) }
	@inlinable public init(_ v: UInt16) { self = FloatToFloat16Rnd(Float(v)) }
	@inlinable public init(_ v: Int16)  { self = FloatToFloat16Rnd(Float(v)) }
	@inlinable public init(_ v: Int32)  { self = FloatToFloat16Rnd(Float(v)) }
	@inlinable public init(_ v: Int)    { self = FloatToFloat16Rnd(Float(v)) }
	@inlinable public init(_ v: Float)  { self = FloatToFloat16Rnd(v) }
	@inlinable public init(_ d: Double) { self = FloatToFloat16Rnd(Float(d)) }
	@inlinable public init(d: Double)   { self = FloatToFloat16Rnd(Float(d)) }
	
    //--------------------------------------------------------------------------
	// functions
	@inlinable public var mantissa: Int { (Int)(x & Float16.mantissaMask) }
	@inlinable public var exponent: Int { (Int)(x & Float16.exponentMask) }
	@inlinable public var sign: Int { (Int)(x & Float16.signMask) }

    @inlinable public static func <(lhs: Float16, rhs: Float16) -> Bool {
		Float(lhs) < Float(rhs)
	}
	
	@inlinable public static func ==(lhs: Float16, rhs: Float16) -> Bool {
		lhs.x == rhs.x
	}
	
	// operators
	@inlinable public static func +(lhs: Float16, rhs: Float16) -> Float16 {
        Float16(Float(lhs) + Float(rhs))
	}
	
	@inlinable public static func -(lhs: Float16, rhs: Float16) -> Float16 {
        Float16(Float(lhs) - Float(rhs))
	}

	@inlinable public static func *(lhs: Float16, rhs: Float16) -> Float16 {
        Float16(Float(lhs) * Float(rhs))
	}

	@inlinable public static func /(lhs: Float16, rhs: Float16) -> Float16 {
        Float16(Float(lhs) / Float(rhs))
	}
}

//==============================================================================
// helpers
@inlinable public func habs(_ h: Float16) -> Float16 {
    Float16(bitPattern: h.x & UInt16(0x7fff))
}

@inlinable public func hneg(_ h: Float16) -> Float16 {
    Float16(bitPattern: h.x ^ UInt16(0x8000))
}

@inlinable public func ishnan(_ h: Float16) -> Bool {
	// When input is NaN, exponent is all 1s and mantissa is non-zero.
    (h.x & UInt16(0x7c00)) == UInt16(0x7c00) && (h.x & UInt16(0x03ff)) != 0
}

@inlinable public func ishinf(_ h: Float16) -> Bool {
	// When input is +/- inf, exponent is all 1s and mantissa is zero.
    (h.x & UInt16(0x7c00)) == UInt16(0x7c00) && (h.x & UInt16(0x03ff)) == 0
}

@inlinable public func ishequ(x: Float16, y: Float16) -> Bool {
    !ishnan(x) && !ishnan(y) && x.x == y.x
}

@inlinable public func hzero() -> Float16 { Float16() }

@inlinable public func hone() -> Float16 { Float16(bitPattern: UInt16(0x3c00)) }

//==============================================================================
// extensions
extension Float {
	@inlinable public init(_ fp16: Float16) { self = Float16ToFloat(fp16) }
}

extension UInt8 {
	@inlinable public init(_ fp16: Float16) { self = UInt8(Float(fp16)) }
}

extension UInt16 {
	@inlinable public init(_ fp16: Float16) { self = UInt16(Float(fp16)) }
}

extension Int16 {
	@inlinable public init(_ fp16: Float16) { self = Int16(Float(fp16)) }
}

extension Int32 {
	@inlinable public init(_ fp16: Float16) { self = Int32(Float(fp16)) }
}

extension Int {
	@inlinable public init(_ fp16: Float16) { self = Int(Float(fp16)) }
}

extension Double {
	@inlinable public init(_ fp16: Float16) { self = Double(Float(fp16)) }
}

//==============================================================================
/// FloatToFloat16Rnd
///	converts from Float to Float16 with rounding
@inlinable public func FloatToFloat16Rnd(_ f: Float) -> Float16 {
	let x = f.bitPattern
	let u: UInt32 = x & 0x7fffffff
	var remainder, shift, lsb, lsb_s1, lsb_m1:UInt32
	var sign, exponent, mantissa: UInt32
	
	// Get rid of +NaN/-NaN case first.
	if (u > 0x7f800000) {
        return Float16(bitPattern: 0x7fff)
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


//==============================================================================
/// Float16ToFloat
/// converts from Float16 to Float
@inlinable public func Float16ToFloat(_ h: Float16) -> Float
{
	var sign     = UInt32((h.x >> 15) & 1)
	var exponent = UInt32((h.x >> 10) & 0x1f)
	var mantissa = UInt32(h.x & 0x3ff) << 13
	
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














