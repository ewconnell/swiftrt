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
	init(any: AnyConvertable)
	init?(string: String)
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
	init(norm any: AnyConvertable)
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
    public init(any: AnyConvertable) { self = any.asInt8 }
    public var asInt8   : Int8   { return self }
    public var asUInt8  : UInt8  { return UInt8(self) }
    public var asUInt16 : UInt16 { return UInt16(self) }
    public var asInt16  : Int16  { return Int16(self) }
    public var asUInt32 : UInt32 { return UInt32(self) }
    public var asInt32  : Int32  { return Int32(self) }
    public var asUInt   : UInt   { return UInt(self) }
    public var asInt    : Int    { return Int(self) }
    public var asFloat  : Float  { return Float(self) }
    public var asDouble : Double { return Double(self) }
    public var asCVarArg: CVarArg{ return self }
    public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }
    
    public init(norm any: AnyConvertable) { self = any.normInt8 }
    public static var normScale: Double = 1.0 / (Double(Int8.max) + 1)
    public static var normScalef: Float = Float(1.0) / (Float(Int8.max) + 1)
    
    public var normInt8   : Int8   { return asInt8 }
    public var normUInt8  : UInt8  { return asUInt8 }
    public var normUInt16 : UInt16 { return asUInt16 }
    public var normInt16  : Int16  { return asInt16 }
    public var normUInt32 : UInt32 { return asUInt32 }
    public var normInt32  : Int32  { return asInt32 }
    public var normUInt   : UInt   { return asUInt }
    public var normInt    : Int    { return asInt }
    public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int8.normScalef }
    public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int8.normScale }
    public var normBool   : Bool   { return asBool }
    
    public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType { return .real8U }
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 4)hhu"
    }
    
    public init?(string: String) {
        guard let value = Int8(string) else { return nil }
        self = value
    }
}

//------------------------------------------------------------------------------
extension UInt8: AnyInteger {
	public init(any: AnyConvertable) { self = any.asUInt8 }
    public var asInt8   : Int8   { return Int8(self) }
	public var asUInt8  : UInt8  { return self }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
    public var asUInt32 : UInt32 { return UInt32(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }

	public init(norm any: AnyConvertable) { self = any.normUInt8 }
	public static var normScale: Double = 1.0 / (Double(UInt8.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt8.max) + 1)
	
    public var normInt8   : Int8   { return asInt8 }
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
    public var normUInt32 : UInt32 { return asUInt32 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt8.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt8.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType { return .real8U }
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 4)hhu"
    }
    
	public init?(string: String) {
        guard let value = UInt8(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension UInt16 : AnyInteger {
	public init(any: AnyConvertable) { self = any.asUInt16 }
    public var asInt8   : Int8   { return Int8(self) }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return self }
	public var asInt16  : Int16  { return Int16(self) }
    public var asUInt32 : UInt32 { return UInt32(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }

	public init(norm any: AnyConvertable) { self = any.normUInt16 }
	public static var normScale: Double = 1.0 / (Double(UInt16.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt16.max) + 1)

    public var normInt8   : Int8   { return asInt8 }
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
    public var normUInt32 : UInt32 { return asUInt32 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt16.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt16.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType { return .real16U }
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)hu"
    }

    public init?(string: String) {
        guard let value = UInt16(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Int16 : AnyInteger {
	public init(any: AnyConvertable) { self = any.asInt16 }
    public var asInt8   : Int8   { return Int8(self) }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return self }
    public var asUInt32 : UInt32 { return UInt32(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }

	public init(norm any: AnyConvertable) { self = any.normInt16 }
	public static var normScale: Double = 1.0 / (Double(Int16.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int16.max) + 1)
	
    public var normInt8   : Int8   { return asInt8 }
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
    public var normUInt32 : UInt32 { return asUInt32 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int16.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int16.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType { return .real16I }
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)hd"
    }

	public init?(string: String) {
        guard let value = Int16(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Int32 : AnyInteger {
	public init(any: AnyConvertable) { self = any.asInt32 }
    public var asInt8   : Int8   { return Int8(self) }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
    public var asUInt32 : UInt32 { return UInt32(self) }
	public var asInt32  : Int32  { return self }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }

	public init(norm any: AnyConvertable) { self = any.normInt32 }
	public static var normScale: Double = 1.0 / (Double(Int32.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int32.max) + 1)

    public var normInt8   : Int8   { return asInt8 }
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
    public var normUInt32 : UInt32 { return asUInt32 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int32.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int32.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType { return .real32I }
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)d"
    }

	public init?(string: String) {
        guard let value = Int32(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension UInt32 : AnyInteger {
    public init(any: AnyConvertable) { self = any.asUInt32 }
    public var asInt8   : Int8   { return Int8(self) }
    public var asUInt8  : UInt8  { return UInt8(self) }
    public var asUInt16 : UInt16 { return UInt16(self) }
    public var asInt16  : Int16  { return Int16(self) }
    public var asUInt32 : UInt32 { return self }
    public var asInt32  : Int32  { return Int32(self) }
    public var asUInt   : UInt   { return UInt(self) }
    public var asInt    : Int    { return Int(self) }
    public var asFloat  : Float  { return Float(self) }
    public var asDouble : Double { return Double(self) }
    public var asCVarArg: CVarArg{ return self }
    public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }

    public init(norm any: AnyConvertable) { self = any.normUInt32 }
    public static var normScale: Double = 1.0 / (Double(UInt32.max) + 1)
    public static var normScalef: Float = Float(1.0) / (Float(UInt32.max) + 1)
    
    public var normInt8   : Int8   { return asInt8 }
    public var normUInt8  : UInt8  { return asUInt8 }
    public var normUInt16 : UInt16 { return asUInt16 }
    public var normInt16  : Int16  { return asInt16 }
    public var normUInt32 : UInt32 { return asUInt32 }
    public var normInt32  : Int32  { return asInt32 }
    public var normUInt   : UInt   { return asUInt }
    public var normInt    : Int    { return asInt }
    public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt32.normScalef }
    public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt32.normScale }
    public var normBool   : Bool   { return asBool }
    
    public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType { return .real32U }
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)u"
    }

    public init?(string: String) {
        guard let value = UInt32(string) else { return nil }
        self = value
    }
}

//------------------------------------------------------------------------------
extension Int : AnyInteger {
	public init(any: AnyConvertable) { self = any.asInt }
    public var asInt8   : Int8   { return Int8(self) }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
    public var asUInt32 : UInt32 { return UInt32(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return self }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }

	public init(norm any: AnyConvertable) { self = any.normInt }
	public static var normScale: Double = 1.0 / (Double(Int.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int.max) + 1)
	
    public var normInt8   : Int8   { return asInt8 }
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
    public var normUInt32 : UInt32 { return asUInt32 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType = {
        let index: [ScalarType] = [.real8I, .real16I, .real32I, .real64I]
        return index[MemoryLayout<Int>.size - 1]
    }()
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 8)d"
    }

	public init?(string: String) {
        guard let value = Int(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension UInt : AnyInteger {
	public init(any: AnyConvertable) { self = any.asUInt }
    public var asInt8   : Int8   { return Int8(self) }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
    public var asUInt32 : UInt32 { return UInt32(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return self }
	public var asInt    : Int    { return Int(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }

	public init(norm any: AnyConvertable) { self = any.normUInt }
	public static var normScale: Double = 1.0 / (Double(UInt.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt.max) + 1)

    public var normInt8   : Int8   { return asInt8 }
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
    public var normUInt32 : UInt32 { return asUInt32 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType = {
        let index: [ScalarType] = [.real8U, .real16U, .real32U, .real64U]
        return index[MemoryLayout<Int>.size - 1]
    }()
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 8)u"
    }

	public init?(string: String) {
        guard let value = UInt(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Bool: AnyConvertable {
	public init(any: AnyConvertable) { self = any.asBool }
    public var asInt8   : Int8   { return self ? 1 : 0 }
	public var asUInt8  : UInt8  { return self ? 1 : 0 }
	public var asUInt16 : UInt16 { return self ? 1 : 0 }
	public var asInt16  : Int16  { return self ? 1 : 0 }
    public var asUInt32 : UInt32 { return self ? 1 : 0 }
	public var asInt32  : Int32  { return self ? 1 : 0 }
	public var asUInt   : UInt   { return self ? 1 : 0 }
	public var asInt    : Int    { return self ? 1 : 0 }
	public var asFloat  : Float  { return self ? 1 : 0 }
	public var asDouble : Double { return self ? 1 : 0 }
	public var asCVarArg: CVarArg{ return self.asInt }
	public var asBool   : Bool   { return self }
	public var asString : String { return self ? "true" : "false" }

	public init(norm any: AnyConvertable) { self = any.normBool }
	public static var normScale: Double = 1
	public static var normScalef : Float = 1

    public var normInt8   : Int8   { return asInt8 }
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
    public var normUInt32 : UInt32 { return asUInt32 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat  : Float  { return Float(any: self) * Bool.normScalef }
	public var normDouble : Double { return Double(any: self) * Bool.normScale}
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var scalarType: ScalarType { return .bool }
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 6)d"
    }

	public init?(string: String) {
        guard let value = Bool(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Float : AnyFloatingPoint {
	public init(any: AnyConvertable) { self = any.asFloat }
    public var asInt8   : Int8   { return Int8(self) }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
    public var asUInt32 : UInt32 { return UInt32(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat  : Float  { return self }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }

	public init(norm any: AnyConvertable) { self = any.normFloat }
	
    public var normInt8   : Int8   { return Int8(Float(self)   * Float(Int8.max))}
	public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
    public var normUInt32 : UInt32 { return UInt32(Float(self) * Float(UInt32.max))}
	public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	public var normFloat  : Float  { return asFloat }
	public var normDouble : Double { return asDouble }
	public var normBool   : Bool   { return asBool }

    public static var normScalef: Float = 1
	public var isFiniteValue: Bool { return self.isFinite }
    public static var isFiniteType: Bool { return false }
    public static var scalarType: ScalarType { return .real32F }
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 9).\(format?.precision ?? 3)f"
    }

	public init?(string: String) {
        guard let value = Float(string) else { return nil }
		self = value
	}
    
    // zero and one (to support Cuda)
    public static var zero: Self = 0
    public static var zeroPointer: UnsafeRawPointer {
        return UnsafeRawPointer(&zero)
    }

    public static var one: Self = 1
    public static var onePointer: UnsafeRawPointer {
        return UnsafeRawPointer(&one)
    }
}

//------------------------------------------------------------------------------
extension Double : AnyFloatingPoint {
	public init(any: AnyConvertable) { self = any.asDouble }
    public var asInt8   : Int8   { return Int8(self) }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
    public var asUInt32 : UInt32 { return UInt32(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return self }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }
    public var asString : String { return String(self) }

	public init(norm any: AnyConvertable) { self = any.normDouble }
	
    public var normInt8   : Int8   { return Int8(Float(self)   * Float(Int8.max))}
	public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
    public var normUInt32 : UInt32 { return UInt32(Float(self) * Float(UInt32.max))}
	public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	public var normFloat  : Float  { return asFloat }
	public var normDouble : Double { return asDouble }
	public var normBool   : Bool   { return asBool }

    public static var normScalef: Float = 1
	public var isFiniteValue: Bool { return self.isFinite }
    public static var isFiniteType: Bool { return false }
    public static var scalarType: ScalarType { return .real64F }
    public static func formatString(_ format: (width: Int, precision: Int)?) -> String {
        return "%\(format?.width ?? 9).\(format?.precision ?? 3)f"
    }

	public init?(string: String) {
        guard let value = Double(string) else { return nil }
		self = value
	}
    
    // zero and one (to support Cuda)
    public static var zero: Self = 0
    public static var zeroPointer: UnsafeRawPointer {
        return UnsafeRawPointer(&zero)
    }
    
    public static var one: Self = 1
    public static var onePointer: UnsafeRawPointer {
        return UnsafeRawPointer(&one)
    }
}

