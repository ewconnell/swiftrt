//******************************************************************************
// Copyright 2019 Google LLC
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
// *** NOTE: There might eventually be no need for AnyConvertable
//
import Foundation
import Complex

//==============================================================================
/// AnyElement
public protocol AnyElement {
    init()
}

public protocol AnyFixedSizeScalar: AnyElement { }

//==============================================================================
/// AnyConvertable
/// AnyNumeric enables the use of constants, type conversion, and
/// normalization within generics
///
public protocol AnyConvertable: AnyFixedSizeScalar, CVarArg {
	// unchanged cast value
	@inlinable init(any: AnyConvertable)
	@inlinable init?(string: String)
    var asInt8   : Int8    { get }
	var asUInt8  : UInt8   { get }
	var asUInt16 : UInt16  { get }
	var asInt16  : Int16   { get }
    var asUInt32 : UInt32  { get }
	var asInt32  : Int32   { get }
	var asUInt   : UInt    { get }
	var asInt    : Int     { get }
	var asFloat  : Float   { get }
	var asDouble : Double  { get }
	var asCVarArg: CVarArg { get }
	var asBool   : Bool    { get }
    var asString : String  { get }

	// values are normalized to the new type during a cast
	@inlinable init(norm any: AnyConvertable)
    var normInt8   : Int8    { get }
	var normUInt8  : UInt8   { get }
	var normUInt16 : UInt16  { get }
	var normInt16  : Int16   { get }
    var normUInt32 : UInt32  { get }
	var normInt32  : Int32   { get }
	var normUInt   : UInt    { get }
	var normInt    : Int     { get }
	var normFloat  : Float   { get }
	var normDouble : Double  { get }
	var normBool   : Bool    { get }

    static var normScalef: Float { get }
	var isFiniteValue: Bool { get }
    static var isFiniteType: Bool { get }
    static var scalarType: ScalarType { get }
    static func formatString(_ format: (width: Int, precision: Int)?) -> String
}

public protocol AnyNumeric: AnyConvertable, Numeric { }
public protocol AnyInteger: BinaryInteger, AnyNumeric {}
public protocol AnyFloatingPoint: FloatingPoint, AnyNumeric {
    static var zeroPointer: UnsafeRawPointer { get }
    static var onePointer: UnsafeRawPointer { get }
}

//------------------------------------------------------------------------------
extension Int8: AnyInteger {
    @inlinable public init(any: AnyConvertable) { self = any.asInt8 }
    @inlinable public var asInt8   : Int8   { return self }
    @inlinable public var asUInt8  : UInt8  { return UInt8(self) }
    @inlinable public var asUInt16 : UInt16 { return UInt16(self) }
    @inlinable public var asInt16  : Int16  { return Int16(self) }
    @inlinable public var asUInt32 : UInt32 { return UInt32(self) }
    @inlinable public var asInt32  : Int32  { return Int32(self) }
    @inlinable public var asUInt   : UInt   { return UInt(self) }
    @inlinable public var asInt    : Int    { return Int(self) }
    @inlinable public var asFloat  : Float  { return Float(self) }
    @inlinable public var asDouble : Double { return Double(self) }
    @inlinable public var asCVarArg: CVarArg{ return self }
    @inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }
    
    @inlinable public init(norm any: AnyConvertable) { self = any.normInt8 }
    public static var normScale: Double = 1.0 / (Double(Int8.max) + 1)
    public static var normScalef: Float = Float(1.0) / (Float(Int8.max) + 1)
    
    @inlinable public var normInt8   : Int8   { return asInt8 }
    @inlinable public var normUInt8  : UInt8  { return asUInt8 }
    @inlinable public var normUInt16 : UInt16 { return asUInt16 }
    @inlinable public var normInt16  : Int16  { return asInt16 }
    @inlinable public var normUInt32 : UInt32 { return asUInt32 }
    @inlinable public var normInt32  : Int32  { return asInt32 }
    @inlinable public var normUInt   : UInt   { return asUInt }
    @inlinable public var normInt    : Int    { return asInt }
    @inlinable public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int8.normScalef }
    @inlinable public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int8.normScale }
    @inlinable public var normBool   : Bool   { return asBool }
    
    @inlinable public var isFiniteValue: Bool { return true }
    @inlinable public static var isFiniteType: Bool { return true }
    @inlinable public static var scalarType: ScalarType { return .real8U }
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 4)hhu"
    }
    
    @inlinable public init?(string: String) {
        guard let value = Int8(string) else { return nil }
        self = value
    }
}

//------------------------------------------------------------------------------
extension UInt8: AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asUInt8 }
    @inlinable public var asInt8   : Int8   { return Int8(self) }
	@inlinable public var asUInt8  : UInt8  { return self }
	@inlinable public var asUInt16 : UInt16 { return UInt16(self) }
	@inlinable public var asInt16  : Int16  { return Int16(self) }
    @inlinable public var asUInt32 : UInt32 { return UInt32(self) }
	@inlinable public var asInt32  : Int32  { return Int32(self) }
	@inlinable public var asUInt   : UInt   { return UInt(self) }
	@inlinable public var asInt    : Int    { return Int(self) }
	@inlinable public var asFloat  : Float  { return Float(self) }
	@inlinable public var asDouble : Double { return Double(self) }
	@inlinable public var asCVarArg: CVarArg{ return self }
	@inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }

	@inlinable public init(norm any: AnyConvertable) { self = any.normUInt8 }
	public static var normScale: Double = 1.0 / (Double(UInt8.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt8.max) + 1)
	
    @inlinable public var normInt8   : Int8   { return asInt8 }
	@inlinable public var normUInt8  : UInt8  { return asUInt8 }
	@inlinable public var normUInt16 : UInt16 { return asUInt16 }
	@inlinable public var normInt16  : Int16  { return asInt16 }
    @inlinable public var normUInt32 : UInt32 { return asUInt32 }
	@inlinable public var normInt32  : Int32  { return asInt32 }
	@inlinable public var normUInt   : UInt   { return asUInt }
	@inlinable public var normInt    : Int    { return asInt }
	@inlinable public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt8.normScalef }
	@inlinable public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt8.normScale }
	@inlinable public var normBool   : Bool   { return asBool }

	@inlinable public var isFiniteValue: Bool { return true }
    @inlinable public static var isFiniteType: Bool { return true }
    @inlinable public static var scalarType: ScalarType { return .real8U }
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 4)hhu"
    }
    
	@inlinable public init?(string: String) {
        guard let value = UInt8(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension UInt16 : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asUInt16 }
    @inlinable public var asInt8   : Int8   { return Int8(self) }
	@inlinable public var asUInt8  : UInt8  { return UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { return self }
	@inlinable public var asInt16  : Int16  { return Int16(self) }
    @inlinable public var asUInt32 : UInt32 { return UInt32(self) }
	@inlinable public var asInt32  : Int32  { return Int32(self) }
	@inlinable public var asUInt   : UInt   { return UInt(self) }
	@inlinable public var asInt    : Int    { return Int(self) }
	@inlinable public var asFloat  : Float  { return Float(self) }
	@inlinable public var asDouble : Double { return Double(self) }
	@inlinable public var asCVarArg: CVarArg{ return self }
	@inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }

	@inlinable public init(norm any: AnyConvertable) { self = any.normUInt16 }
	public static var normScale: Double = 1.0 / (Double(UInt16.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt16.max) + 1)

    @inlinable public var normInt8   : Int8   { return asInt8 }
	@inlinable public var normUInt8  : UInt8  { return asUInt8 }
	@inlinable public var normUInt16 : UInt16 { return asUInt16 }
	@inlinable public var normInt16  : Int16  { return asInt16 }
    @inlinable public var normUInt32 : UInt32 { return asUInt32 }
	@inlinable public var normInt32  : Int32  { return asInt32 }
	@inlinable public var normUInt   : UInt   { return asUInt }
	@inlinable public var normInt    : Int    { return asInt }
	@inlinable public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt16.normScalef }
	@inlinable public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt16.normScale }
	@inlinable public var normBool   : Bool   { return asBool }

	@inlinable public var isFiniteValue: Bool { return true }
    @inlinable public static var isFiniteType: Bool { return true }
    @inlinable public static var scalarType: ScalarType { return .real16U }
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)hu"
    }

    @inlinable public init?(string: String) {
        guard let value = UInt16(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Int16 : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asInt16 }
    @inlinable public var asInt8   : Int8   { return Int8(self) }
	@inlinable public var asUInt8  : UInt8  { return UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { return UInt16(self) }
	@inlinable public var asInt16  : Int16  { return self }
    @inlinable public var asUInt32 : UInt32 { return UInt32(self) }
	@inlinable public var asInt32  : Int32  { return Int32(self) }
	@inlinable public var asUInt   : UInt   { return UInt(self) }
	@inlinable public var asInt    : Int    { return Int(self) }
	@inlinable public var asFloat  : Float  { return Float(self) }
	@inlinable public var asDouble : Double { return Double(self) }
	@inlinable public var asCVarArg: CVarArg{ return self }
	@inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }

	@inlinable public init(norm any: AnyConvertable) { self = any.normInt16 }
	public static var normScale: Double = 1.0 / (Double(Int16.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int16.max) + 1)
	
    @inlinable public var normInt8   : Int8   { return asInt8 }
	@inlinable public var normUInt8  : UInt8  { return asUInt8 }
	@inlinable public var normUInt16 : UInt16 { return asUInt16 }
	@inlinable public var normInt16  : Int16  { return asInt16 }
    @inlinable public var normUInt32 : UInt32 { return asUInt32 }
	@inlinable public var normInt32  : Int32  { return asInt32 }
	@inlinable public var normUInt   : UInt   { return asUInt }
	@inlinable public var normInt    : Int    { return asInt }
	@inlinable public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int16.normScalef }
	@inlinable public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int16.normScale }
	@inlinable public var normBool   : Bool   { return asBool }

	@inlinable public var isFiniteValue: Bool { return true }
    @inlinable public static var isFiniteType: Bool { return true }
    @inlinable public static var scalarType: ScalarType { return .real16I }
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)hd"
    }

	@inlinable public init?(string: String) {
        guard let value = Int16(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Int32 : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asInt32 }
    @inlinable public var asInt8   : Int8   { return Int8(self) }
	@inlinable public var asUInt8  : UInt8  { return UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { return UInt16(self) }
	@inlinable public var asInt16  : Int16  { return Int16(self) }
    @inlinable public var asUInt32 : UInt32 { return UInt32(self) }
	@inlinable public var asInt32  : Int32  { return self }
	@inlinable public var asUInt   : UInt   { return UInt(self) }
	@inlinable public var asInt    : Int    { return Int(self) }
	@inlinable public var asFloat  : Float  { return Float(self) }
	@inlinable public var asDouble : Double { return Double(self) }
	@inlinable public var asCVarArg: CVarArg{ return self }
	@inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }

	@inlinable public init(norm any: AnyConvertable) { self = any.normInt32 }
	public static var normScale: Double = 1.0 / (Double(Int32.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int32.max) + 1)

    @inlinable public var normInt8   : Int8   { return asInt8 }
	@inlinable public var normUInt8  : UInt8  { return asUInt8 }
	@inlinable public var normUInt16 : UInt16 { return asUInt16 }
	@inlinable public var normInt16  : Int16  { return asInt16 }
    @inlinable public var normUInt32 : UInt32 { return asUInt32 }
	@inlinable public var normInt32  : Int32  { return asInt32 }
	@inlinable public var normUInt   : UInt   { return asUInt }
	@inlinable public var normInt    : Int    { return asInt }
	@inlinable public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int32.normScalef }
	@inlinable public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int32.normScale }
	@inlinable public var normBool   : Bool   { return asBool }

	@inlinable public var isFiniteValue: Bool { return true }
    @inlinable public static var isFiniteType: Bool { return true }
    @inlinable public static var scalarType: ScalarType { return .real32I }
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)d"
    }

	@inlinable public init?(string: String) {
        guard let value = Int32(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension UInt32 : AnyInteger {
    @inlinable public init(any: AnyConvertable) { self = any.asUInt32 }
    @inlinable public var asInt8   : Int8   { return Int8(self) }
    @inlinable public var asUInt8  : UInt8  { return UInt8(self) }
    @inlinable public var asUInt16 : UInt16 { return UInt16(self) }
    @inlinable public var asInt16  : Int16  { return Int16(self) }
    @inlinable public var asUInt32 : UInt32 { return self }
    @inlinable public var asInt32  : Int32  { return Int32(self) }
    @inlinable public var asUInt   : UInt   { return UInt(self) }
    @inlinable public var asInt    : Int    { return Int(self) }
    @inlinable public var asFloat  : Float  { return Float(self) }
    @inlinable public var asDouble : Double { return Double(self) }
    @inlinable public var asCVarArg: CVarArg{ return self }
    @inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }

    @inlinable public init(norm any: AnyConvertable) { self = any.normUInt32 }
    public static var normScale: Double = 1.0 / (Double(UInt32.max) + 1)
    public static var normScalef: Float = Float(1.0) / (Float(UInt32.max) + 1)
    
    @inlinable public var normInt8   : Int8   { return asInt8 }
    @inlinable public var normUInt8  : UInt8  { return asUInt8 }
    @inlinable public var normUInt16 : UInt16 { return asUInt16 }
    @inlinable public var normInt16  : Int16  { return asInt16 }
    @inlinable public var normUInt32 : UInt32 { return asUInt32 }
    @inlinable public var normInt32  : Int32  { return asInt32 }
    @inlinable public var normUInt   : UInt   { return asUInt }
    @inlinable public var normInt    : Int    { return asInt }
    @inlinable public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt32.normScalef }
    @inlinable public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt32.normScale }
    @inlinable public var normBool   : Bool   { return asBool }
    
    @inlinable public var isFiniteValue: Bool { return true }
    @inlinable public static var isFiniteType: Bool { return true }
    @inlinable public static var scalarType: ScalarType { return .real32U }
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)u"
    }

    @inlinable public init?(string: String) {
        guard let value = UInt32(string) else { return nil }
        self = value
    }
}

//------------------------------------------------------------------------------
extension Int : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asInt }
    @inlinable public var asInt8   : Int8   { return Int8(self) }
	@inlinable public var asUInt8  : UInt8  { return UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { return UInt16(self) }
	@inlinable public var asInt16  : Int16  { return Int16(self) }
    @inlinable public var asUInt32 : UInt32 { return UInt32(self) }
	@inlinable public var asInt32  : Int32  { return Int32(self) }
	@inlinable public var asUInt   : UInt   { return UInt(self) }
	@inlinable public var asInt    : Int    { return self }
	@inlinable public var asFloat  : Float  { return Float(self) }
	@inlinable public var asDouble : Double { return Double(self) }
	@inlinable public var asCVarArg: CVarArg{ return self }
	@inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }

	@inlinable public init(norm any: AnyConvertable) { self = any.normInt }
	public static var normScale: Double = 1.0 / (Double(Int.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int.max) + 1)
	
    @inlinable public var normInt8   : Int8   { return asInt8 }
	@inlinable public var normUInt8  : UInt8  { return asUInt8 }
	@inlinable public var normUInt16 : UInt16 { return asUInt16 }
	@inlinable public var normInt16  : Int16  { return asInt16 }
    @inlinable public var normUInt32 : UInt32 { return asUInt32 }
	@inlinable public var normInt32  : Int32  { return asInt32 }
	@inlinable public var normUInt   : UInt   { return asUInt }
	@inlinable public var normInt    : Int    { return asInt }
	@inlinable public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int.normScalef }
	@inlinable public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int.normScale }
	@inlinable public var normBool   : Bool   { return asBool }

	@inlinable public var isFiniteValue: Bool { return true }
    @inlinable public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType = {
        let index: [ScalarType] = [.real8I, .real16I, .real32I, .real64I]
        return index[MemoryLayout<Int>.size - 1]
    }()
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 8)d"
    }

	@inlinable public init?(string: String) {
        guard let value = Int(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension UInt : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asUInt }
    @inlinable public var asInt8   : Int8   { return Int8(self) }
	@inlinable public var asUInt8  : UInt8  { return UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { return UInt16(self) }
	@inlinable public var asInt16  : Int16  { return Int16(self) }
    @inlinable public var asUInt32 : UInt32 { return UInt32(self) }
	@inlinable public var asInt32  : Int32  { return Int32(self) }
	@inlinable public var asUInt   : UInt   { return self }
	@inlinable public var asInt    : Int    { return Int(self) }
	@inlinable public var asFloat  : Float  { return Float(self) }
	@inlinable public var asDouble : Double { return Double(self) }
	@inlinable public var asCVarArg: CVarArg{ return self }
	@inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }

	@inlinable public init(norm any: AnyConvertable) { self = any.normUInt }
	public static var normScale: Double = 1.0 / (Double(UInt.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt.max) + 1)

    @inlinable public var normInt8   : Int8   { return asInt8 }
	@inlinable public var normUInt8  : UInt8  { return asUInt8 }
	@inlinable public var normUInt16 : UInt16 { return asUInt16 }
	@inlinable public var normInt16  : Int16  { return asInt16 }
    @inlinable public var normUInt32 : UInt32 { return asUInt32 }
	@inlinable public var normInt32  : Int32  { return asInt32 }
	@inlinable public var normUInt   : UInt   { return asUInt }
	@inlinable public var normInt    : Int    { return asInt }
	@inlinable public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt.normScalef }
	@inlinable public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt.normScale }
	@inlinable public var normBool   : Bool   { return asBool }

	@inlinable public var isFiniteValue: Bool { return true }
    @inlinable public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType = {
        let index: [ScalarType] = [.real8U, .real16U, .real32U, .real64U]
        return index[MemoryLayout<Int>.size - 1]
    }()
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 8)u"
    }

	@inlinable public init?(string: String) {
        guard let value = UInt(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Bool: AnyConvertable {
	@inlinable public init(any: AnyConvertable) { self = any.asBool }
    @inlinable public var asInt8   : Int8   { return self ? 1 : 0 }
	@inlinable public var asUInt8  : UInt8  { return self ? 1 : 0 }
	@inlinable public var asUInt16 : UInt16 { return self ? 1 : 0 }
	@inlinable public var asInt16  : Int16  { return self ? 1 : 0 }
    @inlinable public var asUInt32 : UInt32 { return self ? 1 : 0 }
	@inlinable public var asInt32  : Int32  { return self ? 1 : 0 }
	@inlinable public var asUInt   : UInt   { return self ? 1 : 0 }
	@inlinable public var asInt    : Int    { return self ? 1 : 0 }
	@inlinable public var asFloat  : Float  { return self ? 1 : 0 }
	@inlinable public var asDouble : Double { return self ? 1 : 0 }
	@inlinable public var asCVarArg: CVarArg{ return self.asInt }
	@inlinable public var asBool   : Bool   { return self }
	@inlinable public var asString : String { return self ? "true" : "false" }

	@inlinable public init(norm any: AnyConvertable) { self = any.normBool }
	public static var normScale: Double = 1
	public static var normScalef : Float = 1

    @inlinable public var normInt8   : Int8   { return asInt8 }
	@inlinable public var normUInt8  : UInt8  { return asUInt8 }
	@inlinable public var normUInt16 : UInt16 { return asUInt16 }
	@inlinable public var normInt16  : Int16  { return asInt16 }
    @inlinable public var normUInt32 : UInt32 { return asUInt32 }
	@inlinable public var normInt32  : Int32  { return asInt32 }
	@inlinable public var normUInt   : UInt   { return asUInt }
	@inlinable public var normInt    : Int    { return asInt }
	@inlinable public var normFloat  : Float  { return Float(any: self) * Bool.normScalef }
	@inlinable public var normDouble : Double { return Double(any: self) * Bool.normScale}
	@inlinable public var normBool   : Bool   { return asBool }

	@inlinable public var isFiniteValue: Bool { return true }
    @inlinable public static var isFiniteType: Bool { return true }
    @inlinable public static var scalarType: ScalarType { return .bool }
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)d"
    }

	@inlinable public init?(string: String) {
        guard let value = Bool(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Float : AnyFloatingPoint {
	@inlinable public init(any: AnyConvertable) { self = any.asFloat }
    @inlinable public var asInt8   : Int8   { return Int8(self) }
	@inlinable public var asUInt8  : UInt8  { return UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { return UInt16(self) }
	@inlinable public var asInt16  : Int16  { return Int16(self) }
    @inlinable public var asUInt32 : UInt32 { return UInt32(self) }
	@inlinable public var asInt32  : Int32  { return Int32(self) }
	@inlinable public var asUInt   : UInt   { return UInt(self) }
	@inlinable public var asInt    : Int    { return Int(self) }
	@inlinable public var asFloat  : Float  { return self }
	@inlinable public var asDouble : Double { return Double(self) }
	@inlinable public var asCVarArg: CVarArg{ return self }
	@inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }

	@inlinable public init(norm any: AnyConvertable) { self = any.normFloat }
	
    @inlinable public var normInt8   : Int8   { return Int8(Float(self)   * Float(Int8.max))}
	@inlinable public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	@inlinable public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	@inlinable public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
    @inlinable public var normUInt32 : UInt32 { return UInt32(Float(self) * Float(UInt32.max))}
	@inlinable public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	@inlinable public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	@inlinable public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	@inlinable public var normFloat  : Float  { return asFloat }
	@inlinable public var normDouble : Double { return asDouble }
	@inlinable public var normBool   : Bool   { return asBool }

    public static var normScalef: Float = 1
	@inlinable public var isFiniteValue: Bool { return self.isFinite }
    @inlinable public static var isFiniteType: Bool { return false }
    @inlinable public static var scalarType: ScalarType { return .real32F }
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 9).\(format?.precision ?? 3)f"
    }

	@inlinable public init?(string: String) {
        guard let value = Float(string) else { return nil }
		self = value
	}
    
    // zero and one (to support Cuda)
    public static var zero: Self = 0
    @inlinable public static var zeroPointer: UnsafeRawPointer {
        return UnsafeRawPointer(&zero)
    }

    public static var one: Self = 1
    @inlinable public static var onePointer: UnsafeRawPointer {
        return UnsafeRawPointer(&one)
    }
}

//------------------------------------------------------------------------------
extension Double : AnyFloatingPoint {
	@inlinable public init(any: AnyConvertable) { self = any.asDouble }
    @inlinable public var asInt8   : Int8   { return Int8(self) }
	@inlinable public var asUInt8  : UInt8  { return UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { return UInt16(self) }
	@inlinable public var asInt16  : Int16  { return Int16(self) }
    @inlinable public var asUInt32 : UInt32 { return UInt32(self) }
	@inlinable public var asInt32  : Int32  { return Int32(self) }
	@inlinable public var asUInt   : UInt   { return UInt(self) }
	@inlinable public var asInt    : Int    { return Int(self) }
	@inlinable public var asFloat  : Float  { return Float(self) }
	@inlinable public var asDouble : Double { return self }
	@inlinable public var asCVarArg: CVarArg{ return self }
	@inlinable public var asBool   : Bool   { return self != 0 }
    @inlinable public var asString : String { return String(self) }

	@inlinable public init(norm any: AnyConvertable) { self = any.normDouble }
	
    @inlinable public var normInt8   : Int8   { return Int8(Float(self)   * Float(Int8.max))}
	@inlinable public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	@inlinable public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	@inlinable public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
    @inlinable public var normUInt32 : UInt32 { return UInt32(Float(self) * Float(UInt32.max))}
	@inlinable public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	@inlinable public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	@inlinable public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	@inlinable public var normFloat  : Float  { return asFloat }
	@inlinable public var normDouble : Double { return asDouble }
	@inlinable public var normBool   : Bool   { return asBool }

    public static var normScalef: Float = 1
	@inlinable public var isFiniteValue: Bool { return self.isFinite }
    @inlinable public static var isFiniteType: Bool { return false }
    @inlinable public static var scalarType: ScalarType { return .real64F }
    @inlinable public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 9).\(format?.precision ?? 3)f"
    }

	@inlinable public init?(string: String) {
        guard let value = Double(string) else { return nil }
		self = value
	}
    
    // zero and one (to support Cuda)
    public static var zero: Self = 0
    @inlinable public static var zeroPointer: UnsafeRawPointer {
        return UnsafeRawPointer(&zero)
    }
    
    public static var one: Self = 1
    @inlinable public static var onePointer: UnsafeRawPointer {
        return UnsafeRawPointer(&one)
    }
}

