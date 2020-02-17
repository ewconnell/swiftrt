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

    var buffer: UnsafeBufferPointer<Element> { get }
    var position: Index { get set }
    var shape: Shape { get }
}

//==============================================================================
/// ElementBuffer
public struct ElementBuffer<Element, Shape>: ShapedBuffer
    where Shape: ShapeProtocol
{
    public typealias Index = Int
    public let buffer: UnsafeBufferPointer<Element>
    public var count: Int { shape.count }
    public let endIndex: Index
    public var position: Index
    public let shape: Shape
    public let startIndex: Index

    @inlinable
    public init(_ shape: Shape, _ buffer: UnsafeBufferPointer<Element>) {
        self.shape = shape
        self.buffer = buffer
        startIndex = 0 //Shape.zeros
        endIndex = 0 //shape.extents
        position = startIndex
    }
    
    //-----------------------------------
    // Collection
    @inlinable
    public func index(after i: Index) -> Index {
        fatalError()
    }

    @inlinable
    public subscript(index: Index) -> Element {
        fatalError()
//        buffer[index]
    }
}

//==============================================================================
/// MutableShapedBuffer
public protocol MutableShapedBuffer {
    associatedtype Element
    associatedtype Shape: ShapeProtocol

    var shape: Shape { get }
    var buffer: UnsafeMutableBufferPointer<Element> { get }
}

//==============================================================================
/// MutableElementBuffer
public struct MutableElementBuffer<Element, Shape>: MutableShapedBuffer
    where Shape: ShapeProtocol
{
    public var shape: Shape
    public var buffer: UnsafeMutableBufferPointer<Element>
}

