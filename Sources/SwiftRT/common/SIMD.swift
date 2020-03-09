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

import Foundation

// https://github.com/apple/swift/blob/master/stdlib/public/core/SIMDVectorTypes.swift.gyb

// This isn't actually used to do SIMD operations, but merely as
// a placeholder to satisfy Shape1 Bounds conformance
@frozen public struct SIMD1<Scalar>: SIMD where Scalar: SIMDScalar {
    public var _storage: Scalar.SIMD2Storage
    public typealias MaskStorage = SIMD1<Scalar.SIMDMaskScalar>
    
    /// The number of scalars in the vector.
    @_transparent
    public var scalarCount: Int { 1 }
    
    /// Creates a vector with zero in all lanes.
    @_transparent
    public init() {
        _storage = Scalar.SIMD2Storage()
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

// to support 5D tensors
@frozen public struct SIMD5<Scalar>: SIMD where Scalar: SIMDScalar {
    public var _storage: Scalar.SIMD8Storage
    public typealias MaskStorage = SIMD5<Scalar.SIMDMaskScalar>
    
    /// The number of scalars in the vector.
    @_transparent
    public var scalarCount: Int { 5 }
    
    /// Creates a vector with zero in all lanes.
    @_transparent
    public init() {
        _storage = Scalar.SIMD8Storage()
    }

    @_transparent
    public init(_ v0: Scalar, _ v1: Scalar, _ v2: Scalar, _ v3: Scalar, _ v4: Scalar) {
        self.init()
        self[0] = v0
        self[1] = v1
        self[2] = v2
        self[3] = v3
        self[4] = v4
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
