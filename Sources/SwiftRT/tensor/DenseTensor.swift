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
//
//extension DenseTensor: Differentiable where Element: Differentiable {
//    public typealias TangentVector = Self
//}

//==============================================================================
/// DenseTensor
public struct DenseTensor<Shape, Element>: MutableTensor
    where Shape: TensorShape
{
    // types
    public typealias Index = ElementIndex<Shape>

    // properties
    /// the diagnostic name for the collection
    @inlinable public static var name: String { "DenseTensor\(Shape.rank)" }
    /// the element storage buffer.
    @usableFromInline var storage: StorageBufferType<Element>
    /// the dense number of elements in the shape
    public let elementCount: Int
    /// the storage buffer base offset where this tensor's elements begin
    public let baseOffset: Int
    /// `true` if elements are in row major contiguous order
    public let isSequential: Bool
    /// Specifies whether data is stored in row-major (C-style)
    /// or column-major (Fortran-style) order in memory.
    public let storageOrder: StorageOrder
    /// the dimensions of the element space
    public let shape: Shape
    // used for makeIndex
    @usableFromInline var shapeStrides: Shape { shape.sequentialStrides() }
    /// The strided number of elements spanned by the shape
    public let spanCount: Int
    /// The distance to the next element along each dimension
    public let strides: Shape

    //-----------------------------------
    // device compatibility properties
    @inlinable public var asElement: Element? {
        elementCount == 1 ? storage.element(at: 0) : nil
    }
    
    @inlinable @_transparent
    public var asDense: DenseTensor<Shape, Element> { self }

    //-----------------------------------
    /// `true` if the view will be shared by by multiple writers
    @inlinable public var isShared: Bool { _isShared }
    @usableFromInline var _isShared: Bool

    //-----------------------------------
    /// the starting index zero relative to the storage buffer
    public let startIndex: Index
    /// the ending index zero relative to the storage buffer
    public let endIndex: Index

    //-----------------------------------
    /// defined during init to compute a linear storage buffer index
    @usableFromInline let linear: (Index) -> Int
    /// defined during init to increment an index to the next position
    @usableFromInline let increment: (Index) -> Index

    //--------------------------------------------------------------------------
    // fully specified init
    @inlinable public init(
        shape: Shape,
        strides: Shape,
        elementCount: Int,
        spanCount: Int,
        storage: StorageBufferType<Element>,
        baseOffset: Int,
        order: StorageOrder,
        share: Bool,
        isSequential: Bool
    ) {
        self.shape = shape
        self.strides = strides
        self.elementCount = elementCount
        self.spanCount = spanCount
        self.storage = storage
        self.baseOffset = baseOffset
        self.storageOrder = order
        self._isShared = share
        self.isSequential = isSequential
        self.startIndex = Index(Shape.zero, baseOffset)
        self.endIndex = Index(shape, baseOffset + elementCount)

        //----------------------------------
        // element access functions depending on memory order
        if isSequential {
            linear = { $0.sequencePosition }
            increment = { Index(at: $0.sequencePosition &+ 1) }
        } else {
            linear = { [strides = self.strides] in
                $0.linearIndex(strides)
            }
            increment = { [start = self.startIndex, end = self.endIndex] in
                $0.incremented(between: start, and: end)
            }
        }
    }
}

//==============================================================================
// DenseTensor collection and sub view extensions
public extension DenseTensor {

    /// makeIndex(position:
    /// makes an index from a logical position within `shape`
    /// - Parameters:
    ///  - position: the n-dimensional coordinate position within `shape`
    /// - Returns: the index
    @inlinable func makeIndex(at position: Shape) -> Index {
        Index(position, position.index(stridedBy: shapeStrides))
    }

    /// index(i:
    @inlinable func index(after i: Index) -> Index { increment(i) }

    // elemment subscript
    @inlinable subscript(index: Index) -> Element {
        get {
            storage.element(at: linear(index))
        }
        set {
            storage.setElement(value: newValue, at: linear(index))
        }
    }

    //--------------------------------------------------------------------------
    // view subscripts
    @inlinable subscript(lower: Shape, upper: Shape) -> Self {
        get {
            let shape = upper &- lower
            return DenseTensor(
                shape: shape,
                strides: strides,
                elementCount: shape.elementCount(),
                spanCount: shape.spanCount(stridedBy: strides),
                storage: storage,
                baseOffset: lower.index(stridedBy: strides),
                order: storageOrder,
                share: isShared,
                isSequential: strides == shapeStrides)
        }
        set {
            let shape = upper &- lower
            var view = DenseTensor(
                shape: shape,
                strides: strides,
                elementCount: shape.elementCount(),
                spanCount: shape.spanCount(stridedBy: strides),
                storage: storage,
                baseOffset: lower.index(stridedBy: strides),
                order: storageOrder,
                share: isShared,
                isSequential: strides == shapeStrides)
            
            copy(from: newValue, to: &view)
        }
    }
    
    //--------------------------------------------------------------------------
    /// shared(
    @inlinable mutating func shared() -> Self {
        // if not uniquely held then copy before creating the shared view
        if !isKnownUniquelyReferenced(&storage) {
            diagnostic("\(mutationString) \(storage.name)(\(storage.id)) " +
                "\(Element.self)[\(elementCount)]",
                categories: [.dataCopy, .dataMutation])

            storage = StorageBufferType(copying: storage)
        }
        
        // copy self and set the isShared flag to true
        var sharedDense = self
        sharedDense._isShared = true
        return sharedDense
    }
}

//==============================================================================
// DenseTensor read write extensions
public extension DenseTensor {
    @inlinable func read() {
        
    }
    
    @inlinable func read(using queue: DeviceQueue) {
    }

    @inlinable mutating func readWrite() {
    }
    
    @inlinable mutating func readWrite(using queue: DeviceQueue) {
    }
}

//==============================================================================
// DenseTensor initializers
public extension DenseTensor {
    /// init(shape:order:
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

    /// init(element:shape:order:
    @inlinable init(_ element: Element, _ shape: Shape,
                    order: StorageOrder = .C)
    {
        self.init(shape, order: order)

        // initialize storage on the target device
        copy(from: RepeatedElement(shape, element: element), to: &self)
    }
    
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element == Element
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        _ = storage.hostBuffer.initialize(from: elements)
    }
    
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
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
    
    /// init(shape:order:elements:
    /// implicitly casts from C.Element float -> Element integer
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryInteger
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
    
    /// init(shape:order:elements:
    /// implicitly casts from C.Element float -> Element integer
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
}
