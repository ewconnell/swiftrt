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
    // properties
    public typealias Index = ElementIndex<Shape>
    @inlinable public static var name: String { "RepeatedElement\(Shape.rank)" }
    public let elementCount: Int
    public let shape: Shape
    public let shapeStrides: Shape
    public let storageOrder: StorageOrder
    public let element: Element

    // Collection properties
    public let startIndex: Index
    public let endIndex: Index

    //-----------------------------------
    // device compatibility properties
    @inlinable  @_transparent
    public var asElement: Element? { element }
    
    @inlinable @_transparent
    public var asDense: DenseTensor<Shape, Element> {
        DenseTensor(element, Shape.one, order: .C)
    }

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
        shapeStrides = shape.sequentialStrides()
        elementCount = shape.elementCount()
        startIndex = Index(at: 0)
        endIndex = Index(at: elementCount)
    }
    
    //------------------------------------
    // Collection functions
    @inlinable public func elements() -> Self { self }
    @inlinable public subscript(index: Index) -> Element { element }
    @inlinable public func index(after i: Index) -> Index {
        Index(at: i.sequencePosition + 1)
    }

    //------------------------------------
    // view subscripts
    @inlinable public subscript(lower: Shape, upper: Shape) -> Self {
        RepeatedElement(upper &- lower, element: element, order: storageOrder)
    }
}

//------------------------------------------------------------------------------
// extensions
extension RepeatedElement: Equatable where Element: Equatable { }
extension RepeatedElement: Codable where Element: Codable { }

//==============================================================================
/// EyeTensor
public struct EyeTensor<Element>: Tensor, Collection
    where Element: Numeric
{
    // properties
    public typealias Index = ElementIndex<Shape2>
    @inlinable public static var name: String { "EyeTensor" }
    public let elementCount: Int
    public let shape: Shape2
    public let shapeStrides: Shape2
    public let storageOrder: StorageOrder
    public let k: Int

    // Collection properties
    public let startIndex: Index
    public let endIndex: Index

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
        self.shapeStrides = self.shape.sequentialStrides()
        self.storageOrder = order
        self.k = k
        self.elementCount = shape.elementCount()
        self.startIndex = Index(lower, 0)
        self.endIndex = Index(upper, self.elementCount)
    }

    //-----------------------------------
    // device compatibility properties
    @inlinable  @_transparent
    public var asElement: Element? {
        fatalError()
    }
    
    @inlinable @_transparent
    public var asDense: DenseTensor<Shape, Element> {
        fatalError()
    }

    //------------------------------------
    // Collection functions
    @inlinable public func elements() -> Self { self }
    @inlinable public func index(after i: Index) -> Index {
        fatalError()
    }

    @inlinable public subscript(index: Index) -> Element {
        // if the axes indexes are equal then it's on the diagonal
        let pos = index.position // &- k
        return pos[0] == pos[1] ? 1 : 0
    }

    //------------------------------------
    // view subscripts
    @inlinable public subscript(lower: Shape2, upper: Shape2) -> Self {
        EyeTensor(from: lower, to: upper, k: k, storage: storageOrder)
    }
}

//------------------------------------------------------------------------------
// extensions
extension EyeTensor: Equatable where Element: Equatable { }
extension EyeTensor: Codable where Element: Codable { }
