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
/// RepeatedElement
/// Repeats an element value for all indices
public struct RepeatedElement<Shape, Element>: Tensor, Collection
    where Shape: TensorShape
{
    // Tensor properties
    @inlinable public static var name: String { "RepeatedElement\(Shape.rank)" }
    public let elementCount: Int
    public let shape: Shape
    public let storageOrder: StorageOrder
    public let element: Element

    // Collection properties
    public let startIndex: Int
    public let endIndex: Int

    //------------------------------------
    /// init(shape:element:order:
    /// - Parameters:
    ///  - shape: the shape of the tensor
    ///  - element: the element value to repeat
    ///  - order: the order in memory to store materialized Elements
    @inlinable public init(
        _ shape: Shape,
        element: Element,
        order: StorageOrder = .rowMajor
    ) {
        self.shape = shape
        self.element = element
        self.storageOrder = order
        elementCount = shape.elementCount()
        startIndex = 0
        endIndex = elementCount
    }
    
    //------------------------------------
    // Collection functions
    @inlinable public func elements() -> Self { self }
    @inlinable public subscript(index: Int) -> Element { element }
    @inlinable public func index(after i: Int) -> Int { i + 1 }

    //------------------------------------
    // view subscripts
    @inlinable public subscript(lower: Shape, upper: Shape) -> Self {
        RepeatedElement(upper &- lower, element: element, order: storageOrder)
    }
    
    @inlinable public subscript(lower: Shape, upper: Shape, steps: Shape) -> Self {
        fatalError()
    }
}

//------------------------------------------------------------------------------
// extensions
extension RepeatedElement: Equatable where Element: Equatable { }
extension RepeatedElement: Codable where Element: Codable { }

//==============================================================================
/// IndexTensor
/// A collection where each element is equal to it's index
public struct IndexTensor<Shape, Element>: Tensor, Collection
    where Shape: TensorShape, Element: Numeric
{
    // Tensor properties
    @inlinable public static var name: String { "IndexTensor\(Shape.rank)" }
    public let elementCount: Int
    public let shape: Shape
    public let storageOrder: StorageOrder

    // Collection properties
    public let startIndex: Int
    public let endIndex: Int

    //------------------------------------
    /// init(lower:upper:order:
    /// - Parameters:
    ///  - lower: lower bound of the range
    ///  - upper: upper bound of the range
    ///  - order: the order in memory to store materialized Elements
    @inlinable public init(
        from lower: Shape,
        to upper: Shape,
        order: StorageOrder = .rowMajor
    ) {
        self.shape = upper &- lower
        self.storageOrder = order
        elementCount = shape.elementCount()
        startIndex = lower.elementCount()
        endIndex = startIndex + elementCount
    }
    
    //------------------------------------
    // Collection functions
    @inlinable public func elements() -> Self { self }
    @inlinable public func index(after i: Int) -> Int { i + 1 }
    @inlinable public subscript(index: Int) -> Element {
        assert(Element(exactly: index) != nil,
               "index value is too large for Element type")
        return Element(exactly: index)!
    }

    //------------------------------------
    // view subscripts
    @inlinable public subscript(lower: Shape, upper: Shape) -> Self {
        IndexTensor(from: lower, to: upper, order: storageOrder)
    }
    
    @inlinable
    public subscript(lower: Shape, upper: Shape, steps: Shape) -> Self {
        IndexTensor(from: lower, to: upper, order: storageOrder)
    }
}

//------------------------------------------------------------------------------
// extensions
extension IndexTensor: Equatable where Element: Equatable { }
extension IndexTensor: Codable where Element: Codable { }

//==============================================================================
/// EyeTensor
public struct EyeTensor<Element>: Tensor, Collection
    where Element: Numeric
{
    // tensor properties
    @inlinable public static var name: String { "EyeTensor" }
    public let elementCount: Int
    public let shape: Shape2
    public let storageOrder: StorageOrder
    public let k: Int

    // Collection properties
    public let startIndex: StridedIndex<Shape2>
    public let endIndex: StridedIndex<Shape2>

    //------------------------------------
    /// init(lower:upper:order:
    /// - Parameters:
    ///  - lower: lower bound of the range
    ///  - upper: upper bound of the range
    ///  - k:
    ///  - order: the order in memory to store materialized Elements
    @inlinable public init(
        from lower: Shape2,
        to upper: Shape2,
        k: Int,
        storage order: StorageOrder = .rowMajor
    ) {
        self.shape = upper &- lower
        self.storageOrder = order
        self.k = k
        self.elementCount = shape.elementCount()
        self.startIndex = Index(lower, 0)
        self.endIndex = Index(upper, self.elementCount)
    }
    
    //------------------------------------
    // Collection functions
    @inlinable public func elements() -> Self { self }
    @inlinable public func index(after i: StridedIndex<Shape2>)
        -> StridedIndex<Shape2>
    {
        i.incremented(boundedBy: shape)
    }

    @inlinable public subscript(index: StridedIndex<Shape2>) -> Element {
        // if the axes indexes are equal then it's on the diagonal
        let pos = index.position // &- k
        return pos[0] == pos[1] ? 1 : 0
    }

    //------------------------------------
    // view subscripts
    @inlinable public subscript(lower: Shape2, upper: Shape2) -> Self {
        EyeTensor(from: lower, to: upper, k: k, storage: storageOrder)
    }
    
    @inlinable
    public subscript(lower: Shape2, upper: Shape2, steps: Shape2) -> Self {
        EyeTensor(from: lower, to: upper, k: k, storage: storageOrder)
    }
}

//------------------------------------------------------------------------------
// extensions
extension EyeTensor: Equatable where Element: Equatable { }
extension EyeTensor: Codable where Element: Codable { }
