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
public protocol Tensor: Logging, CustomStringConvertible {
    //----------------------------------
    /// the ranked short vector type that defines the tensor coordinate space
    associatedtype Shape: Shaped
    /// the type of element in the collection
    associatedtype Element
    /// the type used to iterate the elements
    associatedtype Elements: Collection
        where Elements.Element == Element

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
    
    //----------------------------------
    /// Returns a collection of elements (stored or generated)
    /// `Tensor` does not conform to `Collection` directly. The `elements`
    /// function first ensures synchronization for read only access.
    func elements() -> Elements
    
    /// subscript
    /// returns a sub view
    /// - Parameters:
    ///  - position: the view starting point
    ///  - shape: shape of the view
    subscript(position: Shape, shape: Shape) -> Self { get }

    /// subscript
    /// returns a sub view
    /// - Parameters:
    ///  - position: the view starting point
    ///  - shape: shape of the view
    ///  - steps: steps across the parent space
    subscript(position: Shape, shape: Shape, steps: Shape) -> Self { get }
}

//==============================================================================
/// MutableTensor
/// an n-dimensional mutable collection of stored elements
public protocol MutableTensor: Tensor {
    //----------------------------------
    /// tye type of element storage buffer
    associatedtype Buffer: StorageBuffer where Buffer.Element == Element
    /// a type used to iterate the elements
    associatedtype MutableElements: MutableCollection
        where MutableElements.Element == Element

    //----------------------------------
    /// class reference to the underlying platform element buffer
    var storageBuffer: Buffer { get }
    /// the linear element offset where the view begins
    var bufferOffset: Int { get }
    /// `true` if the view will be shared by by multiple writers
    var isShared: Bool { get }

    //----------------------------------
    /// Returns a mutable collection of stored elements
    /// `MutableTensor` does not conform to `MutableCollection` directly.
    /// The `mutableElements` function first ensures synchronization
    /// for read write access.
    /// - Parameters:
    ///  - willOverwrite: `true` indicates that the caller will overwrite
    ///    all elements in the collection, which eliminates the need to
    ///    read data back from discreet memory space if in use
    /// - Returns: a mutable collection of elements
    func mutableElements(willOverwrite: Bool) -> MutableElements
    
    /// shared
    /// returns a sub view that does not do copy-on-write to allow
    /// for multi-threaded writes.
    /// - Parameters:
    ///  - position: the view starting point
    ///  - shape: shape of the view
    ///  - steps: steps across the parent space
    func shared(at position: Shape, with shape: Shape, by steps: Shape?) -> Self

    /// subscript
    /// returns a sub view
    /// - Parameters:
    ///  - position: the view starting point
    ///  - shape: shape of the view
    subscript(position: Shape, shape: Shape) -> Self { get set }

    /// subscript
    /// returns a sub view
    /// - Parameters:
    ///  - position: the view starting point
    ///  - shape: shape of the view
    ///  - steps: steps across the parent space
    subscript(position: Shape, shape: Shape, steps: Shape) -> Self { get set }
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

