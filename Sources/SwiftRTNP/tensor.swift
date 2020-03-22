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
import SwiftRT

//==============================================================================
// Tensor
public struct Tensor<Bounds, Element>: TensorView
    where Bounds: ShapeBounds
{
    // properties
    public static var diagnosticName: String { "Tensor\(Bounds.rank)" }
    public let shape: Shape<Bounds>
    public var buffer: TensorBuffer<Element>
    public let offset: Int
    public let shared: Bool

    @inlinable
    public init(shape: Shape<Bounds>, buffer: TensorBuffer<Element>,
                offset: Int, shared: Bool)
    {
        self.shape = shape
        self.buffer = buffer
        self.offset = offset
        self.shared = shared
    }
}

//==============================================================================
// Tensor
public extension Tensor {
    //--------------------------------------------------------------------------
    /// reserved space
    @inlinable
    init(bounds: Bounds) {
        self = Self.create(Shape(bounds))
    }

    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    init(repeating value: Element, to bounds: Bounds.Tuple)
    {
        let shape = Shape(Bounds(bounds), strides: Bounds.zero)
        self = Self.create(for: value, shape)
    }

    //--------------------------------------------------------------------------
    // typed views
    @inlinable
    func createBoolTensor(with bounds: Bounds) -> Tensor<Bounds, Bool> {
        Tensor<Bounds, Bool>(bounds: bounds)
    }
    
    @inlinable
    func createIndexTensor(with bounds: Bounds) -> Tensor<Bounds, IndexType> {
        Tensor<Bounds, IndexType>(bounds: bounds)
    }
}

//==============================================================================
// Tensor1
public extension Tensor where Bounds == Bounds1
{
}
