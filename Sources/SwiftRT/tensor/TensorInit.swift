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
// Tensor initializers
public extension Tensor {
    //--------------------------------------------------------------------------
    /// init(shape:order:
    /// creates a dense shape
    /// - Parameters:
    ///  - shape: the n-dimensional shape of the tensor
    ///  - order: the storage order of the elements
    @inlinable init(_ shape: Shape, order: StorageOrder = .C) {
        let count = shape.elementCount()
        self.init(shape: shape,
                  strides: shape.sequentialStrides(),
                  elementCount: count,
                  spanCount: count,
                  storage: StorageBufferType(count: count, name: Self.name),
                  baseOffset: 0,
                  order: order,
                  share: false,
                  isSequential: true)
    }

    //--------------------------------------------------------------------------
    /// init(like:order:
    /// convenience initializer to initialize with the shape and type as `other`
    /// - Parameters:
    ///  - other: a tensor to copy attributes from
    ///  - order: the storage order of the elements
    @inlinable init<T: TensorType>(like other: T, order: StorageOrder = .C)
        where Shape == T.Shape, Element == T.Element
    {
        self.init(other.shape, order: order)
    }
    
    //--------------------------------------------------------------------------
    /// init(like:order:
    /// convenience initializer to initialize with the shape as `other`.
    /// - Parameters:
    ///  - other: a tensor to copy attributes from
    ///  - order: the storage order of the elements
    @inlinable init<T: TensorType>(like other: T, order: StorageOrder = .C)
        where Shape == T.Shape
    {
        self.init(other.shape, order: order)
    }

    //--------------------------------------------------------------------------
    /// init(element:
    /// creates a tensor with a single scalar value
    /// - Parameter element: the single element value for the tensor
    @inlinable init(_ element: Element) {
        self.init(shape: Shape.one,
                  strides: Shape.one,
                  elementCount: 1,
                  spanCount: 1,
                  storage: StorageBufferType(count: 1, name: Self.name),
                  baseOffset: 0,
                  order: .C,
                  share: false,
                  isSequential: true)
        storage.setElement(value: element, at: 0)
    }

    //--------------------------------------------------------------------------
    /// init(repeating element:shape:
    /// Repeats a single stored element while indexing
    /// - Parameters:
    ///  - element: the element value to repeat while indexing
    ///  - shape: the shape of the tensor
    @inlinable init(repeating element: Element, to shape: Shape) {
        self.init(shape: shape,
                  strides: Shape.zero,
                  elementCount: shape.elementCount(),
                  spanCount: 1,
                  storage: StorageBufferType(count: 1, name: Self.name),
                  baseOffset: 0,
                  order: .C,
                  share: false,
                  isSequential: true)
        storage.setElement(value: element, at: 0)
    }

    //--------------------------------------------------------------------------
    /// init(repeating other:shape:
    /// Repeats a tensor withing the specified shape while indexing
    /// - Parameters:
    ///  - other: the tensor to repeat
    ///  - shape: the shape of the tensor
    @inlinable init(repeating other: Self, to shape: Shape) {
        // make sure the bounds are compatible
        assert({
            for i in 0..<Shape.rank {
                if other.shape[i] != 1 && shape[i] != other.shape[i] {
                    return false
                }
            }
            return true
        }(), "repeated tensor dimensions must be either 1" +
            " or match the repeated shape")

        // compute strides, setting stride to 0 for repeated dimensions
        var repeatedStrides = Shape.zero
        for i in 0..<Shape.rank where other.shape[i] == shape[i] {
            repeatedStrides[i] = other.strides[i]
        }
        
        self.init(shape: shape,
                  strides: repeatedStrides,
                  elementCount: shape.elementCount(),
                  spanCount: shape.spanCount(stridedBy: repeatedStrides),
                  storage: other.storage,
                  baseOffset: other.baseOffset,
                  order: other.storageOrder,
                  share: other.isShared,
                  isSequential: false)
    }
    
    //--------------------------------------------------------------------------
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
    /// - Parameters:
    ///  - elements: the collection used to initialize storage
    ///  - shape: the shape of the tensor
    ///  - order: the storage order
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element == Element
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        _ = storage.hostBuffer.initialize(from: elements)
    }
    
    //--------------------------------------------------------------------------
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
    /// - Parameters:
    ///  - elements: the collection used to initialize storage
    ///  - shape: the shape of the tensor
    ///  - order: the storage order
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element: BinaryInteger, Element: Numeric
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        let lazyElements = elements.lazy.map { value -> Element in
            assert(Element(exactly: value) != nil,
                   "Value cast \(Element.self)(\(value)) failed")
            return Element(exactly: value)!
        }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
    
    //--------------------------------------------------------------------------
    /// init(shape:order:elements:
    /// implicitly casts from C.Element float -> Element integer
    /// - Parameters:
    ///  - elements: the collection used to initialize storage
    ///  - shape: the shape of the tensor
    ///  - order: the storage order
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection,
        C.Element: BinaryFloatingPoint, Element: BinaryInteger
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
    
    //--------------------------------------------------------------------------
    /// init(shape:order:elements:
    /// implicitly casts from C.Element float -> Element integer
    /// - Parameters:
    ///  - elements: the collection used to initialize storage
    ///  - shape: the shape of the tensor
    ///  - order: the storage order
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection,
        C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
    
    //--------------------------------------------------------------------------
    /// reductionBounds
    /// returns the upper bounds for a reduction result along the specified axes
    @inlinable func reductionShape(alongAxes axes: Set<Int>?) -> Shape {
        guard let axes = axes else { return Shape.one }
        assert(axes.isSubset(of: 0..<Shape.rank), "axis is out of bounds")
        var result = shape
        axes.forEach { result[$0] = 1 }
        return result
    }
}

