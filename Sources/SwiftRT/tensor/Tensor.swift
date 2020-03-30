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

//==============================================================================
/// Tensor protocol
/// an n-dimensional collection of elements
public protocol Tensor:
    Collection, CustomStringConvertible, Logging
{
    //----------------------------------
    /// the ranked short vector type that defines the tensor coordinate space
    associatedtype Shape: TensorShape
    /// the type of element in the collection
    associatedtype Element

    //----------------------------------
    /// the number of elements described by `shape`
    var elementCount: Int { get }
    /// a label for the type used as a default name in diagnostics
    static var name: String { get }
    /// the order in memory to store materialized Elements. Generator
    /// tensor types maintain this property as a template for dense
    /// result tensors.
    var storageOrder: StorageOrder { get }
    /// the dimension of the coordinate space
    var shape: Shape { get }
    
    init<Parent, Index>(
        _ parent: Parent,
        from lower: Shape,
        to upper: Shape,
        by steps: Shape,
        indexedBy: Index.Type) where
        Parent: Tensor, Parent.Shape == Shape, Parent.Element == Element,
        Index: TensorIndex, Index.Shape == Shape
    
    /// subscript
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    /// - Returns: an n-dimensional tensor slice
    subscript(lower: Shape, upper: Shape) -> Self { get }
    
    /// subscript
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    ///  - steps: steps along axes in parent space while forming the result
    /// - Returns: an n-dimensional tensor slice
    subscript(lower: Shape, upper: Shape, steps: Shape) -> Self { get }
}

//==============================================================================
/// MutableTensor
/// an n-dimensional mutable collection of stored elements
public protocol MutableTensor: Tensor, MutableCollection {
    //----------------------------------
    /// tye type of element storage buffer
    associatedtype Buffer: StorageBuffer where Buffer.Element == Element

    /// `true` if the view will be shared by by multiple writers
    var isShared: Bool { get }
    /// a linear buffer of materialized `Elements`
    var storageBuffer: Buffer { get }
    
    /// shared
    /// returns a sub view that does not do copy-on-write to allow
    /// for multi-threaded writes.
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    /// - Returns: an n-dimensional tensor slice
    func shared(from lower: Shape, to upper: Shape) -> Self

    /// subscript
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    /// - Returns: an n-dimensional tensor slice
    subscript(lower: Shape, upper: Shape) -> Self { get set }
    
    /// subscript
    /// - Parameters:
    ///  - lower: the lower bound of the slice
    ///  - upper: the upper bound of the slice
    ///  - steps: steps along axes in parent space while forming the result
    /// - Returns: an n-dimensional tensor slice
    subscript(lower: Shape, upper: Shape, steps: Shape) -> Self { get set }
}

//==============================================================================
// default types
/// the type used for memory indexing on discreet devices
public typealias DeviceIndex = Int32

//==============================================================================
/// StorageOrder
/// Specifies how to store multi-dimensional data in row-major (C-style)
/// or column-major (Fortran-style) order in memory.
/// These names are following the numpy naming convention
public enum StorageOrder: Int, Codable {
    /// C style row major memory layout
    case C
    /// Fortran style column major memory layout
    case F
    /// more expressive aliases
    public static let rowMajor = C, colMajor = F
}

