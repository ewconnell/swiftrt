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
import Foundation

//==============================================================================
// ShapeBounds
public protocol ShapeBounds: SIMD where Scalar == Int
{
    /// the number of bounding dimensions
    static var rank: Int { get }
    
    // **** Adding this to the protocol causes about a 40% loss in
    // indexing performance to both Shape indexing and to Normal SIMD
    // operations.
    // Extending the SIMD classes seems to be a really bad thing
    mutating func increment(boundedBy bounds: Self)
    
    func sequentialStrides() -> Self
}

//==============================================================================
// ShapeBounds extensions
public extension ShapeBounds {
    /// instance member access
    @inlinable @_transparent
    var count: Int { Self.rank }

    /// helper
    @inlinable @_transparent
    mutating func swapAt(_ a: Int, _ b: Int) {
        let tmp = self[a]
        self[a] = self[b]
        self[b] = tmp
    }
    
    @inlinable
    mutating func increment(boundedBy bounds: Self) {
        var dim = Self.rank - 1
        while true {
            self[dim] += 1
            if self[dim] < bounds[dim] {
                break
            } else if dim > 0 {
                self[dim] = 0
                dim -= 1
            } else {
                break
            }
        }
    }

    //--------------------------------------------------------------------------
    /// `reduce`
    @inlinable
    func reduce(
        into initialResult: Scalar,
        _ updateAccumulatingResult: (inout Scalar, Scalar) -> ()) -> Scalar
    {
        indices.reduce(into: initialResult) {
            updateAccumulatingResult(&$0, self[$1])
        }
    }

    //--------------------------------------------------------------------------
    /// `sequentialStrides`
    /// computes the row major sequential strides
    @inlinable
    func sequentialStrides() -> Self {
        var strides = Self.one
        for i in stride(from: Self.rank - 1, through: 1, by: -1) {
            strides[i - 1] = self[i] * strides[i]
        }
        return strides
    }
}

//==============================================================================
// SIMD1
extension SIMD1: ShapeBounds where Scalar == Int {
    @inlinable @_transparent
    public static var rank: Int { 1 }
    
    @inlinable
    public func elementCount() -> Int {
        self[0]
    }

    @inlinable
    public func sequentialStrides() -> Self {
        Self(1)
    }
    
    @inlinable
    public mutating func increment(boundedBy bounds: Self) {
        self[0] += 1
    }
}

//==============================================================================
// SIMD2
extension SIMD2: ShapeBounds where Scalar == Int {
    @inlinable @_transparent
    public static var rank: Int { 2 }
    
    @inlinable
    public func elementCount() -> Int {
        self[0] * self[1]
    }

    @inlinable
    public func sequentialStrides() -> Self {
        Self(self[1], 1)
    }

    @inlinable
    public mutating func increment(boundedBy bounds: Self) {
        // a recursive algorithm was ~55x slower
        self[1] += 1

        if self[1] == bounds[1] {
            self[1] = 0
            self[0] += 1
        }
    }
}

//==============================================================================
// SIMD3
extension SIMD3: ShapeBounds where Scalar == Int {
    @inlinable @_transparent
    public static var rank: Int { 3 }
    
    @inlinable
    public mutating func increment(boundedBy bounds: Self) {
        // a recursive algorithm was ~55x slower
        self[2] += 1
        if self[2] == bounds[2] {
            self[2] = 0
            self[1] += 1
            
            if self[1] == bounds[1] {
                self[1] = 0
                self[0] += 1
            }
        }
    }
}

//==============================================================================
// SIMD4
extension SIMD4: ShapeBounds where Scalar == Int {
    @inlinable @_transparent
    public static var rank: Int { 4 }

    @inlinable
    public mutating func increment(boundedBy bounds: Self) {
        // a recursive algorithm was ~55x slower
        self[3] += 1
        if self[3] == bounds[3] {
            self[3] = 0
            self[2] += 1
            
            if self[2] == bounds[2] {
                self[2] = 0
                self[1] += 1
                
                if self[1] == bounds[1] {
                    self[1] = 0
                    self[0] += 1
                }
            }
        }
    }
}

//==============================================================================
// SIMD5
extension SIMD5: ShapeBounds where Scalar == Int {
    @inlinable @_transparent
    public static var rank: Int { 5 }

    @inlinable
    public mutating func increment(boundedBy bounds: Self) {
        // a recursive algorithm was ~55x slower
        self[4] += 1
        if self[4] == bounds[4] {
            self[4] = 0
            self[3] += 1
            
            if self[3] == bounds[3] {
                self[3] = 0
                self[2] += 1
                
                if self[2] == bounds[2] {
                    self[2] = 0
                    self[1] += 1
                    
                    if self[1] == bounds[1] {
                        self[1] = 0
                        self[0] += 1
                    }
                }
            }
        }
    }
}

