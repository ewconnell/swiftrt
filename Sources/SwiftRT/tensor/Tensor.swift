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

// Sequence
// https://www.hackingwithswift.com/example-code/language/how-to-make-a-custom-sequence

//==============================================================================
/// Tensor protocol
/// an n-dimensional collection of elements
public protocol Tensor: Logging {
    /// a ranked type that describes the dimension of the coordinate space
    associatedtype Shape: Shaped
    /// the type of element in the collection
    associatedtype Element
    /// a type used to iterate the elements
    associatedtype ElementSequence: Sequence & IteratorProtocol
    
    /// the dense number of elements specified by shape
    var count: Int { get }
    /// a label for the type used as a default name in diagnostics
    static var name: String { get }
    /// the order in memory to store materialized Elements
    var order: StorageOrder { get }
    /// the dimension of the coordinate space
    var shape: Shape { get }
    
    /// returns a sequence of elements, which might be stored or generated
    func elements() -> ElementSequence
}

//==============================================================================
/// MutableTensor
/// This is used to perform indexed writes to the collection
public protocol MutableTensor: Tensor {
    /// tye type of element storage buffer
    associatedtype Buffer: StorageBuffer where Buffer.Element == Element
    /// a type used to iterate the elements
    associatedtype MutableElements: MutableCollection

    /// class reference to the underlying platform element buffer
    var buffer: Buffer { get }
    /// the linear element offset where the view begins
    var offset: Int { get }
    /// `true` if the view will be shared by by multiple writers
    var shared: Bool { get }

    /// returns an indexed mutable collection of stored elements
    func mutableElements() -> MutableElements
}

//==============================================================================
// default types
/// the type used for indexing on discreet devices
public typealias DeviceIndex = Int32

//==============================================================================
/// StorageOrder
/// Specifies how to store multi-dimensional data in row-major (C-style)
/// or column-major (Fortran-style) order in memory.
public enum StorageOrder: Int, Codable {
    case C, F
    public static let rowMajor = C, colMajor = F
}

