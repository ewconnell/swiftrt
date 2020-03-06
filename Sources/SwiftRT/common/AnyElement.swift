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
import Numerics

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
    @inlinable public init?(string: String) {
        guard let value = Int8(string) else { return nil }
        self = value
    }

    @inlinable public var asInt8   : Int8   { self }
    @inlinable public var asUInt8  : UInt8  { UInt8(self) }
    @inlinable public var asUInt16 : UInt16 { UInt16(self) }
    @inlinable public var asInt16  : Int16  { Int16(self) }
    @inlinable public var asUInt32 : UInt32 { UInt32(self) }
    @inlinable public var asInt32  : Int32  { Int32(self) }
    @inlinable public var asUInt   : UInt   { UInt(self) }
    @inlinable public var asInt    : Int    { Int(self) }
    @inlinable public var asFloat  : Float  { Float(self) }
    @inlinable public var asDouble : Double { Double(self) }
    @inlinable public var asCVarArg: CVarArg{ self }
    @inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }
}

//------------------------------------------------------------------------------
extension UInt8: AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asUInt8 }
    @inlinable public init?(string: String) {
        guard let value = UInt8(string) else { return nil }
        self = value
    }

    @inlinable public var asInt8   : Int8   { Int8(self) }
	@inlinable public var asUInt8  : UInt8  { self }
	@inlinable public var asUInt16 : UInt16 { UInt16(self) }
	@inlinable public var asInt16  : Int16  { Int16(self) }
    @inlinable public var asUInt32 : UInt32 { UInt32(self) }
	@inlinable public var asInt32  : Int32  { Int32(self) }
	@inlinable public var asUInt   : UInt   { UInt(self) }
	@inlinable public var asInt    : Int    { Int(self) }
	@inlinable public var asFloat  : Float  { Float(self) }
	@inlinable public var asDouble : Double { Double(self) }
	@inlinable public var asCVarArg: CVarArg{ self }
	@inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }
}

//------------------------------------------------------------------------------
extension UInt16 : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asUInt16 }
    @inlinable public init?(string: String) {
        guard let value = UInt16(string) else { return nil }
        self = value
    }

    @inlinable public var asInt8   : Int8   { Int8(self) }
	@inlinable public var asUInt8  : UInt8  { UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { self }
	@inlinable public var asInt16  : Int16  { Int16(self) }
    @inlinable public var asUInt32 : UInt32 { UInt32(self) }
	@inlinable public var asInt32  : Int32  { Int32(self) }
	@inlinable public var asUInt   : UInt   { UInt(self) }
	@inlinable public var asInt    : Int    { Int(self) }
	@inlinable public var asFloat  : Float  { Float(self) }
	@inlinable public var asDouble : Double { Double(self) }
	@inlinable public var asCVarArg: CVarArg{ self }
	@inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }
}

//------------------------------------------------------------------------------
extension Int16 : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asInt16 }
    @inlinable public init?(string: String) {
        guard let value = Int16(string) else { return nil }
        self = value
    }

    @inlinable public var asInt8   : Int8   { Int8(self) }
	@inlinable public var asUInt8  : UInt8  { UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { UInt16(self) }
	@inlinable public var asInt16  : Int16  { self }
    @inlinable public var asUInt32 : UInt32 { UInt32(self) }
	@inlinable public var asInt32  : Int32  { Int32(self) }
	@inlinable public var asUInt   : UInt   { UInt(self) }
	@inlinable public var asInt    : Int    { Int(self) }
	@inlinable public var asFloat  : Float  { Float(self) }
	@inlinable public var asDouble : Double { Double(self) }
	@inlinable public var asCVarArg: CVarArg{ self }
	@inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }
}

//------------------------------------------------------------------------------
extension Int32 : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asInt32 }
    @inlinable public init?(string: String) {
        guard let value = Int32(string) else { return nil }
        self = value
    }

    @inlinable public var asInt8   : Int8   { Int8(self) }
	@inlinable public var asUInt8  : UInt8  { UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { UInt16(self) }
	@inlinable public var asInt16  : Int16  { Int16(self) }
    @inlinable public var asUInt32 : UInt32 { UInt32(self) }
	@inlinable public var asInt32  : Int32  { self }
	@inlinable public var asUInt   : UInt   { UInt(self) }
	@inlinable public var asInt    : Int    { Int(self) }
	@inlinable public var asFloat  : Float  { Float(self) }
	@inlinable public var asDouble : Double { Double(self) }
	@inlinable public var asCVarArg: CVarArg{ self }
	@inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }
}

//------------------------------------------------------------------------------
extension UInt32 : AnyInteger {
    @inlinable public init(any: AnyConvertable) { self = any.asUInt32 }
    @inlinable public init?(string: String) {
        guard let value = UInt32(string) else { return nil }
        self = value
    }

    @inlinable public var asInt8   : Int8   { Int8(self) }
    @inlinable public var asUInt8  : UInt8  { UInt8(self) }
    @inlinable public var asUInt16 : UInt16 { UInt16(self) }
    @inlinable public var asInt16  : Int16  { Int16(self) }
    @inlinable public var asUInt32 : UInt32 { self }
    @inlinable public var asInt32  : Int32  { Int32(self) }
    @inlinable public var asUInt   : UInt   { UInt(self) }
    @inlinable public var asInt    : Int    { Int(self) }
    @inlinable public var asFloat  : Float  { Float(self) }
    @inlinable public var asDouble : Double { Double(self) }
    @inlinable public var asCVarArg: CVarArg{ self }
    @inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }
}

//------------------------------------------------------------------------------
extension Int : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asInt }
    @inlinable public init?(string: String) {
        guard let value = Int(string) else { return nil }
        self = value
    }

    @inlinable public var asInt8   : Int8   { Int8(self) }
	@inlinable public var asUInt8  : UInt8  { UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { UInt16(self) }
	@inlinable public var asInt16  : Int16  { Int16(self) }
    @inlinable public var asUInt32 : UInt32 { UInt32(self) }
	@inlinable public var asInt32  : Int32  { Int32(self) }
	@inlinable public var asUInt   : UInt   { UInt(self) }
	@inlinable public var asInt    : Int    { self }
	@inlinable public var asFloat  : Float  { Float(self) }
	@inlinable public var asDouble : Double { Double(self) }
	@inlinable public var asCVarArg: CVarArg{ self }
	@inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }
}

//------------------------------------------------------------------------------
extension UInt : AnyInteger {
	@inlinable public init(any: AnyConvertable) { self = any.asUInt }
    @inlinable public init?(string: String) {
        guard let value = UInt(string) else { return nil }
        self = value
    }

    @inlinable public var asInt8   : Int8   { Int8(self) }
	@inlinable public var asUInt8  : UInt8  { UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { UInt16(self) }
	@inlinable public var asInt16  : Int16  { Int16(self) }
    @inlinable public var asUInt32 : UInt32 { UInt32(self) }
	@inlinable public var asInt32  : Int32  { Int32(self) }
	@inlinable public var asUInt   : UInt   { self }
	@inlinable public var asInt    : Int    { Int(self) }
	@inlinable public var asFloat  : Float  { Float(self) }
	@inlinable public var asDouble : Double { Double(self) }
	@inlinable public var asCVarArg: CVarArg{ self }
	@inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }
}

//------------------------------------------------------------------------------
extension Bool: AnyConvertable {
	@inlinable public init(any: AnyConvertable) { self = any.asBool }
    @inlinable public init?(string: String) {
        guard let value = Bool(string) else { return nil }
        self = value
    }

    @inlinable public var asInt8   : Int8   { self ? 1 : 0 }
	@inlinable public var asUInt8  : UInt8  { self ? 1 : 0 }
	@inlinable public var asUInt16 : UInt16 { self ? 1 : 0 }
	@inlinable public var asInt16  : Int16  { self ? 1 : 0 }
    @inlinable public var asUInt32 : UInt32 { self ? 1 : 0 }
	@inlinable public var asInt32  : Int32  { self ? 1 : 0 }
	@inlinable public var asUInt   : UInt   { self ? 1 : 0 }
	@inlinable public var asInt    : Int    { self ? 1 : 0 }
	@inlinable public var asFloat  : Float  { self ? 1 : 0 }
	@inlinable public var asDouble : Double { self ? 1 : 0 }
	@inlinable public var asCVarArg: CVarArg{ self.asInt }
	@inlinable public var asBool   : Bool   { self }
	@inlinable public var asString : String { self ? "true" : "false" }
}

//------------------------------------------------------------------------------
extension Float : AnyFloatingPoint {
	@inlinable public init(any: AnyConvertable) { self = any.asFloat }
    @inlinable public init?(string: String) {
        guard let value = Float(string) else { return nil }
        self = value
    }
    
    @inlinable public var asInt8   : Int8   { Int8(self) }
	@inlinable public var asUInt8  : UInt8  { UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { UInt16(self) }
	@inlinable public var asInt16  : Int16  { Int16(self) }
    @inlinable public var asUInt32 : UInt32 { UInt32(self) }
	@inlinable public var asInt32  : Int32  { Int32(self) }
	@inlinable public var asUInt   : UInt   { UInt(self) }
	@inlinable public var asInt    : Int    { Int(self) }
	@inlinable public var asFloat  : Float  { self }
	@inlinable public var asDouble : Double { Double(self) }
	@inlinable public var asCVarArg: CVarArg{ self }
	@inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }

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
    @inlinable public init?(string: String) {
        guard let value = Double(string) else { return nil }
        self = value
    }
    
    @inlinable public var asInt8   : Int8   { Int8(self) }
	@inlinable public var asUInt8  : UInt8  { UInt8(self) }
	@inlinable public var asUInt16 : UInt16 { UInt16(self) }
	@inlinable public var asInt16  : Int16  { Int16(self) }
    @inlinable public var asUInt32 : UInt32 { UInt32(self) }
	@inlinable public var asInt32  : Int32  { Int32(self) }
	@inlinable public var asUInt   : UInt   { UInt(self) }
	@inlinable public var asInt    : Int    { Int(self) }
	@inlinable public var asFloat  : Float  { Float(self) }
	@inlinable public var asDouble : Double { self }
	@inlinable public var asCVarArg: CVarArg{ self }
	@inlinable public var asBool   : Bool   { self != 0 }
    @inlinable public var asString : String { String(self) }

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

