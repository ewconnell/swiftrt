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

public protocol VectorElement: SIMD & StorageElement where
    Scalar: SIMDScalar & StorageElement { }

////==============================================================================
//// RGB
//public protocol RGBProtocol: SIMD {
//    var r: Scalar { get set }
//    var g: Scalar { get set }
//    var b: Scalar { get set }
//}
//
//extension SIMD3: VectorElement & StorageElement where Scalar: StorageElement {
//    public typealias Stored = Self
//    public typealias Value = Self
//}
//
//public typealias RGB2<Scalar> = SIMD3<Scalar> where
//    Scalar: SIMDScalar & StorageElement
//
//extension RGB2: RGBProtocol {
//    @_transparent public var r: Scalar { get { self[0] } set(v) { self[0] = v }}
//    @_transparent public var g: Scalar { get { self[1] } set(v) { self[1] = v }}
//    @_transparent public var b: Scalar { get { self[2] } set(v) { self[2] = v }}
//
//    @_transparent public init(_ v0: Scalar, _ v1: Scalar, _ v2: Scalar) {
//        self.init()
//        self[0] = v0
//        self[1] = v1
//        self[2] = v2
//    }
//}
//
////==============================================================================
//// RGB
//// TODO!!: need to implement SIMD3Storage so that gpu memory is packed
//@frozen public struct RGB<Scalar>: VectorElement
//where Scalar: SIMDScalar & StorageElement
//{
//    public typealias Stored = Self
//    public typealias Value = Self
//
//    public var _storage: Scalar.SIMD4Storage
//    public typealias MaskStorage = SIMD4<Scalar.SIMDMaskScalar>
//
//    /// The number of scalars in the vector.
//    @_transparent public var scalarCount: Int { 3 }
//    @_transparent public var r: Scalar { get { self[0] } set(v) { self[0] = v }}
//    @_transparent public var g: Scalar { get { self[1] } set(v) { self[1] = v }}
//    @_transparent public var b: Scalar { get { self[2] } set(v) { self[2] = v }}
//
//    /// Creates a pixel with zero in all lanes
//    @_transparent public init() {
//        _storage = Scalar.SIMD4Storage()
//    }
//
//    @_transparent public init(_ v0: Scalar, _ v1: Scalar, _ v2: Scalar) {
//        self.init()
//        self[0] = v0
//        self[1] = v1
//        self[2] = v2
//    }
//
//    /// Accesses the scalar at the specified position.
//    public subscript(index: Int) -> Scalar {
//        @_transparent get {
//            assert(indices.contains(index))
//            return _storage[index]
//        }
//        @_transparent set {
//            assert(indices.contains(index))
//            _storage[index] = newValue
//        }
//    }
//}
//
//extension RGB: AdditiveArithmetic where Scalar: FloatingPoint { }
//
//==============================================================================
// RGBA
@frozen public struct RGBA<Scalar>: VectorElement
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
    
    @_transparent
    public init(_ v0: Scalar, _ v1: Scalar, _ v2: Scalar, _ v3: Scalar) {
        self.init()
        self[0] = v0
        self[1] = v1
        self[2] = v2
        self[3] = v3
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

extension RGBA: AdditiveArithmetic where Scalar: FloatingPoint { }

//==============================================================================
// Stereo
@frozen public struct Stereo<Scalar>: VectorElement
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
