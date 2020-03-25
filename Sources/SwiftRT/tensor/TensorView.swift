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

public typealias DeviceIndex = Int32

////==============================================================================
///// TensorView protocol
///// A TensorView object is the primary interface for working with data.
///// Specialized shaped instances such as Vector, Matrix, Volume, etc..
///// conform to this protocol
/////
//public protocol TensorView: Logging {
//    /// tensor shape
//    associatedtype Bounds: ShapeBounds
//    /// tye type of element storage buffer
//    associatedtype Buffer: StorageBuffer where Buffer.Element == Element
//    /// the type of element stored by the tensor
//    associatedtype Element
//    /// A concrete type used in generics to pass Boolean values
//    associatedtype BoolView: TensorView
//        where BoolView.Element == Bool, BoolView.Bounds == Bounds
//    /// A concrete type used in generics to return index results
//    associatedtype IndexView: TensorView
//        where IndexView.Element == IndexType, IndexView.Bounds == Bounds
//
//    //--------------------------------------------------------------------------
//    // properties
//    /// a label for the type used as a default name in diagnostics
//    static var diagnosticName: String { get }
//    /// the shape of the view used for indexing
//    var shape: TensorShape<Bounds> { get }
//    /// class reference to the underlying platform element buffer
//    var buffer: Buffer { get set }
//    /// the linear element offset where the view begins
//    var offset: Int { get }
//    /// `true` if the view will be shared by by multiple writers
//    var shared: Bool { get }
//
//    //--------------------------------------------------------------------------
//    /// fully specified used for creating tensors
//    init(shape: TensorShape<Bounds>, buffer: Buffer, offset: Int, shared: Bool)
//
//    //--------------------------------------------------------------------------
//    /// creates a new dense tensor of the same type with the specified bounds
//    func createDense(with bounds: Bounds) -> Self
//    /// creates a new dense tensor where `Element` equals `Bool`
//    /// with the specified bounds
//    func createBoolTensor(with bounds: Bounds) -> BoolView
//    /// creates a new dense tensor where `Element` equals `IndexType`
//    /// with the specified bounds and initial values
//    func createIndexTensor(with bounds: Bounds) -> IndexView
//}
