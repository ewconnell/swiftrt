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
/// ShapedBuffer
public protocol ShapedBuffer: Collection {
    associatedtype Element
    associatedtype Shape: ShapeProtocol

    var pointer: UnsafeBufferPointer<Element> { get }
    var shape: Shape { get }
}

//==============================================================================
/// BufferElements
public struct BufferElements<Element, Shape>: ShapedBuffer
    where Shape: ShapeProtocol
{
    public typealias Index = Shape.Index
    public let pointer: UnsafeBufferPointer<Element>
    public let shape: Shape

    @inlinable public var endIndex: Index { shape.endIndex }
    @inlinable public var startIndex: Index { shape.startIndex }

    //-----------------------------------
    // initializers
    @inlinable
    public init(_ shape: Shape, _ pointer: UnsafeBufferPointer<Element>) {
        self.shape = pointer.count > 0 ? shape : Shape(extents: Shape.zeros)
        self.pointer = pointer
    }
    
    //-----------------------------------
    // Collection
    @inlinable @inline(__always)
    public func index(after i: Index) -> Index { shape.index(after: i) }

    @inlinable
    public subscript(index: Index) -> Element {
        pointer[shape[index]]
    }
}

//==============================================================================
/// MutableShapedBuffer
public protocol MutableShapedBuffer: MutableCollection {
    associatedtype Element
    associatedtype Shape: ShapeProtocol

    var pointer: UnsafeMutableBufferPointer<Element> { get }
    var shape: Shape { get }
}

//==============================================================================
/// MutableBufferElements
public struct MutableBufferElements<Element, Shape>: MutableShapedBuffer
    where Shape: ShapeProtocol
{
    public typealias Index = Shape.Index
    public let pointer: UnsafeMutableBufferPointer<Element>
    public let shape: Shape
    
    @inlinable public var endIndex: Index { shape.endIndex }
    @inlinable public var startIndex: Index { shape.startIndex }
    
    //-----------------------------------
    // initializers
    @inlinable
    public init(_ shape: Shape, _ pointer: UnsafeMutableBufferPointer<Element>){
        self.shape = pointer.count > 0 ? shape : Shape(extents: Shape.zeros)
        self.pointer = pointer
    }
    
    //-----------------------------------
    // Collection
    @inlinable @inline(__always)
    public func index(after i: Index) -> Index { shape.index(after: i) }

    @inlinable
    public subscript(index: Index) -> Element {
        get {
            pointer[shape[index]]
        }
        set {
            pointer[shape[index]] = newValue
        }
    }
}

