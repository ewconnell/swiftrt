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
import Foundation
import Numerics

//==============================================================================
/// VectorElement
/// Conforming types can be used as short vector tensor elements
///
/// Comments
/// At this time, an `RGB` vector element is not implemented because
/// we need to determine how memory alignment would work and if it is
/// sensible. With a byte sized `Scalar` like `UInt8`, we want to be
/// 32 bit aligned, so `scalarCount == 4` is required.
/// With a 16 bit `Scalar` like `Float16` or `BFloat16`,
/// `scalarCount == 4` is still required for 32 bit alignment.
/// With a 32 bit `Scalar` like `Float`, the scalars are already aligned so
/// an RGB order would be desirable to minimize memory overhead on the
/// cpu and gpu. On the cpu, it does not appear that llvm suppports
/// a packed 3 element `SIMD` vector (need to fully investigate this).
///
public protocol VectorElement: StorageElement {
    associatedtype Scalar: StorageElement
    
    /// The number of scalars, or elements, in a vector of this type.
    static var scalarCount: Int { get }
}

extension VectorElement {
    @inlinable public static var type: StorageElementType { fatalError("not implemented") }

    @inlinable public static var storedZeroPointer: UnsafeRawPointer {
        fatalError("not implemented")
    }
    
    @inlinable public static var storedOnePointer: UnsafeRawPointer {
        fatalError("not implemented")
    }
}

//==============================================================================
// RGBA
@frozen public struct RGBA<Scalar>: VectorElement, SIMD
where Scalar: SIMDScalar & StorageElement
{
    public typealias Stored = Self
    public typealias Value = Self
    
    public var _storage: Scalar.SIMD4Storage
    public typealias MaskStorage = SIMD4<Scalar.SIMDMaskScalar>
    
    /// The number of scalars in the vector.
    @_transparent public var scalarCount: Int { 4 }
    @_transparent public var r: Scalar { get { self[0] } set(v) { self[0] = v }}
    @_transparent public var g: Scalar { get { self[1] } set(v) { self[1] = v }}
    @_transparent public var b: Scalar { get { self[2] } set(v) { self[2] = v }}
    @_transparent public var a: Scalar { get { self[3] } set(v) { self[3] = v }}



    /// Creates a pixel with zero in all lanes
    @_transparent public init() {
        _storage = Scalar.SIMD4Storage()
    }
    
    /// Accesses the scalar at the specified position.
    public subscript(index: Int) -> Scalar {
        @_transparent get {
            assert(indices.contains(index))
            return _storage[index]
        }
        @_transparent set {
            assert(indices.contains(index))
            _storage[index] = newValue
        }
    }
}

//------------------------------------------------------------------------------
extension RGBA where Scalar: FixedWidthInteger {
    @_transparent public init(
        _ r: Scalar,
        _ g: Scalar,
        _ b: Scalar,
        _ a: Scalar = Scalar.max
    ) {
        self.init()
        self[0] = r
        self[1] = g
        self[2] = b
        self[3] = a
    }
}

extension RGBA where Scalar: BinaryFloatingPoint {
    @_transparent public init(
        _ r: Scalar,
        _ g: Scalar,
        _ b: Scalar,
        _ a: Scalar = 1
    ) {
        self.init()
        self[0] = r
        self[1] = g
        self[2] = b
        self[3] = a
    }
}

extension RGBA: AdditiveArithmetic where Scalar: FloatingPoint { }

//------------------------------------------------------------------------------
@usableFromInline var _storedZeroRGBAFloat32 = RGBA<Float>()
@usableFromInline var _storedOneRGBAFloat32 = RGBA<Float>(1, 1, 1, 1)

public extension VectorElement where Scalar == Float {
    @inlinable static var type: StorageElementType { .vector32Fx4 }

    @inlinable static var storedZeroPointer: UnsafeRawPointer {
        UnsafeRawPointer(&_storedZeroRGBAFloat32) 
    }
    
    @inlinable static var storedOnePointer: UnsafeRawPointer {
        UnsafeRawPointer(&_storedOneRGBAFloat32)
    }
}

//------------------------------------------------------------------------------
@usableFromInline var _storedZeroRGBAUInt8 = RGBA<UInt8>()
@usableFromInline var _storedOneRGBAUInt8 = RGBA<UInt8>(1, 1, 1, 1)

public extension VectorElement where Scalar == UInt8 {
    @inlinable static var type: StorageElementType { .vector8Ux4 }

    @inlinable static var storedZeroPointer: UnsafeRawPointer {
        UnsafeRawPointer(&_storedZeroRGBAUInt8) 
    }
    
    @inlinable static var storedOnePointer: UnsafeRawPointer {
        UnsafeRawPointer(&_storedOneRGBAUInt8)
    }
}

//==============================================================================
// Stereo
@frozen public struct Stereo<Scalar>: VectorElement, SIMD
where Scalar: SIMDScalar & StorageElement
{
    public typealias Stored = Self
    public typealias Value = Self
    
    public var _storage: Scalar.SIMD2Storage
    public typealias MaskStorage = SIMD2<Scalar.SIMDMaskScalar>
    
    /// The number of scalars in the vector.
    @_transparent public var scalarCount: Int { 2 }
    @_transparent public var r: Scalar { get { self[0] } set(v) { self[0] = v }}
    @_transparent public var l: Scalar { get { self[1] } set(v) { self[1] = v }}
    
    /// Creates a pixel with zero in all lanes
    @_transparent public init() {
        _storage = Scalar.SIMD2Storage()
    }
    
    @_transparent public init(_ v0: Scalar, _ v1: Scalar) {
        self.init()
        self[0] = v0
        self[1] = v1
    }
    
    /// Accesses the scalar at the specified position.
    public subscript(index: Int) -> Scalar {
        @_transparent get {
            assert(indices.contains(index))
            return _storage[index]
        }
        @_transparent set {
            assert(indices.contains(index))
            _storage[index] = newValue
        }
    }
}
