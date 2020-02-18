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

    var bufferPointer: UnsafePointer<Element> { get }
    var shape: Shape { get }
}

//==============================================================================
/// ElementBuffer
public struct ElementBuffer<Element, Shape>: ShapedBuffer
    where Shape: ShapeProtocol
{
    public typealias Index = Int
    public let bufferPointer: UnsafePointer<Element>
    public var count: Int { shape.count }
    public let endIndex: Index
    public let shape: Shape
    public let startIndex: Index

    @inlinable
    public init(_ shape: Shape, _ buffer: UnsafePointer<Element>) {
        self.shape = shape
        self.bufferPointer = buffer
        startIndex = 0 //Shape.zeros
        endIndex = 0 //shape.extents
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
public protocol MutableShapedBuffer: MutableCollection {
    associatedtype Element
    associatedtype Shape: ShapeProtocol

    var bufferPointer: UnsafeMutablePointer<Element> { get }
    var shape: Shape { get }
}

//==============================================================================
/// MutableElementBuffer
public struct MutableElementBuffer<Element, Shape>: MutableShapedBuffer
    where Shape: ShapeProtocol
{
    public typealias Index = Int
    public var bufferPointer: UnsafeMutablePointer<Element>
    public var count: Int { shape.count }
    public let endIndex: Index
    public let shape: Shape
    public let startIndex: Index
    
    @inlinable
    public init(_ shape: Shape, _ buffer: UnsafeMutablePointer<Element>) {
        self.shape = shape
        self.bufferPointer = buffer
        startIndex = 0 //Shape.zeros
        endIndex = 0 //shape.extents
    }
    
    //-----------------------------------
    // Collection
    @inlinable
    public func index(after i: Index) -> Index {
        fatalError()
    }
    
    @inlinable
    public subscript(index: Index) -> Element {
        get { fatalError() }
        set { fatalError() }
    }
}

