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
/// FillTensor
public struct FillTensor<Shape, Element>: Tensor, Collection
    where Shape: Shaped
{
    // Tensor properties
    @inlinable public static var name: String { "FillTensor\(Shape.rank)" }
    public typealias Index = Int
    public let elementCount: Int
    public let shape: Shape
    public let storageOrder: StorageOrder
    public let element: Element

    // Collection properties
    public let startIndex: Index
    public let endIndex: Index

    //------------------------------------
    /// init(shape:element:order:
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
    @inlinable public subscript(index: Index) -> Element { element }
    @inlinable public func index(after i: Index) -> Index { i + 1 }

    //------------------------------------
    // view subscripts
    @inlinable public subscript(lower: Shape, upper: Shape) -> Self {
        FillTensor(upper &- lower, element: element, order: storageOrder)
    }
    
    @inlinable public subscript(lower: Shape, upper: Shape, steps: Shape) -> Self {
        fatalError()
    }
}

//------------------------------------------------------------------------------
// extensions
extension FillTensor: Equatable where Element: Equatable { }
extension FillTensor: Codable where Element: Codable { }

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
    public let startIndex: ShapeIndex<Shape2>
    public let endIndex: ShapeIndex<Shape2>

    //------------------------------------
    /// init(shape:k:order:
    @inlinable public init(
        _ shape: Shape2,
        _ k: Int,
        _ order: StorageOrder = .rowMajor
    ) {
        self.k = k
        self.shape = shape
        self.elementCount = shape.elementCount()
        self.storageOrder = order
        self.startIndex = Index(Shape2.zero, 0)
        self.endIndex = Index(self.shape, self.elementCount)
    }
    
    //------------------------------------
    // Collection functions
    @inlinable public func elements() -> Self { self }
    @inlinable public func index(after i: ShapeIndex<Shape2>)
        -> ShapeIndex<Shape2>
    {
        i.incremented(boundedBy: shape)
    }

    @inlinable public subscript(index: ShapeIndex<Shape2>) -> Element {
        // if the axes indexes are equal then it's on the diagonal
        let pos = index.position // &- k
        return pos[0] == pos[1] ? 1 : 0
    }

    //------------------------------------
    // view subscripts
    @inlinable public subscript(lower: Shape2, upper: Shape2) -> Self {
        fatalError()
    }
    
    @inlinable public subscript(lower: Shape2, upper: Shape2, steps: Shape2) -> Self {
        fatalError()
    }
}

//------------------------------------------------------------------------------
// extensions
extension EyeTensor: Equatable where Element: Equatable { }
extension EyeTensor: Codable where Element: Codable { }

