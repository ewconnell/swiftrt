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
/// DenseTensor
public struct DenseTensor<Shape, Element, Index>:
    MutableTensor, MutableCollection
    where
    Index: TensorIndex,
    Index.Shape == Shape
{
    public let storageBuffer: TensorBuffer<Element>
    /// the dense number of elements in the shape
    public let elementCount: Int
    /// the linear element offset where the view begins
    public let bufferOffset: Int
    /// Specifies whether data is stored in row-major (C-style)
    /// or column-major (Fortran-style) order in memory.
    public let storageOrder: StorageOrder
    /// the dimensions of the element space
    public let shape: Shape
    /// `true` if the view will be shared by by multiple writers
    public let isShared: Bool
    /// The strided number of elements spanned by the shape
    public let spanCount: Int
    /// The distance to the next element along each dimension
    public let strides: Shape

    @inlinable public static var name: String { "DenseTensor\(Shape.rank)" }
    @inlinable public var startIndex: Index { Index(Shape.zero, 0) }
    @inlinable public var endIndex: Index { Index(shape, elementCount) }

    //-----------------------------------
    /// init(shape:
    @inlinable
    public init(
        _ shape: Shape,
        storageBuffer: TensorBuffer<Element>? = nil,
        strides: Shape? = nil,
        bufferOffset: Int = 0,
        share: Bool = false,
        order: StorageOrder = .rowMajor,
        element value: Element? = nil
    ) {
        let elementCount = shape.elementCount()
        self.storageBuffer = storageBuffer ??
            TensorBuffer(count: elementCount, name: Self.name, element: value)
        self.elementCount = elementCount
        self.bufferOffset = bufferOffset
        self.storageOrder = order
        self.shape = shape
        self.isShared = share
        let sequentialStrides = shape.sequentialStrides()

        if let callerStrides = strides {
            self.strides = callerStrides
            self.spanCount =  ((shape &- 1) &* callerStrides).wrappedSum() + 1
        } else {
            self.strides = sequentialStrides
            self.spanCount = elementCount
        }
    }
    
    //--------------------------------------------------------------------------
    @inlinable public func index(after i: Index) -> Index {
        i.incremented(boundedBy: shape)
    }
    
    @inlinable public subscript(position: Index) -> Element {
        get {
            fatalError()
        }
        set(newValue) {
            fatalError()
        }
    }
    
    //--------------------------------------------------------------------------
    //
    @inlinable public func shared(from lower: Shape, to upper: Shape) -> Self {
        fatalError()
    }
    
    //--------------------------------------------------------------------------
    // view subscripts
    @inlinable public subscript(lower: Shape, upper: Shape) -> Self {
        get {
            fatalError()
        }
        set {
            fatalError()
        }
    }
    
    @inlinable public subscript(lower: Shape, upper: Shape, steps: Shape) -> Self {
        get {
            fatalError()
        }
        set {
            fatalError()
        }
    }
}

