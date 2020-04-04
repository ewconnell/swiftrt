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
    /// the strides used to compute logical positions within `shape`
    public let shapeStrides: Shape
    /// The strided number of elements spanned by the shape
    public let spanCount: Int
    /// The distance to the next element along each dimension
    public let strides: Shape

    //-----------------------------------
    /// `true` if the view will be shared by by multiple writers
    @inlinable public var isShared: Bool { _isShared }
    @usableFromInline var _isShared: Bool

    //-----------------------------------
    /// the starting index zero relative to the storage buffer
    @inlinable public var startIndex: Index {
        Index(Shape.zero, baseOffset)
    }
    /// the ending index zero relative to the storage buffer
    @inlinable public var endIndex: Index {
        Index(shape, baseOffset + elementCount)
    }

    //-----------------------------------
    /// a function defined during initialization to get storage element index
    @usableFromInline let linear: (Index) -> Int
    
    //--------------------------------------------------------------------------
    /// init(lower:upper:storage:strides:share:order
    @inlinable public init(
        shape: Shape,
        storage: StorageBufferType<Element>? = nil,
        offset: Int = 0,
        strides: Shape? = nil,
        share: Bool = false,
        order: StorageOrder = .rowMajor
    ) {
        self.shape = shape
        self.baseOffset = offset
        self.shapeStrides = shape.sequentialStrides()
        let count = shape.elementCount()
        self.elementCount = count
        self.storageOrder = order
        self._isShared = share
        self.storage = storage ?? StorageBufferType(count: count, name: Self.name)
        
        if let strides = strides {
            self.strides = strides
            self.spanCount = shape.spanCount(with: strides)
            var sequentialStrides = self.shapeStrides
            if shape[0] == 1 { sequentialStrides[0] = strides[0] }
            self.isSequential = strides == sequentialStrides
        } else {
            self.isSequential = true
            self.strides = self.shapeStrides
            self.spanCount = elementCount
        }

        //----------------------------------
        // element access functions depending on memory order
        if isSequential {
            linear = { $0.sequencePosition }
        } else {
            linear = { [offset = baseOffset, strides = self.strides] in
                offset + $0.linearIndex(strides)
            }
        }
    }
}

//==============================================================================
// DenseTensor collection and sub view extensions
public extension DenseTensor {
    //--------------------------------------------------------------------------
    // Collection
    /// index(i:
    @inlinable func index(after i: Index) -> Index {
        i.incremented(between: startIndex, and: endIndex)
    }

    // elemment subscript
    @inlinable subscript(index: Index) -> Element {
        get {
            storage.element(at: linear(index))
        }
        set {
            storage.element(at: linear(index), value: newValue)
        }
    }

    //--------------------------------------------------------------------------
    // view subscripts
    @inlinable subscript(lower: Shape, upper: Shape) -> Self {
        get {
            DenseTensor(shape: upper &- lower, storage: storage,
                        offset: lower.index(stridedBy: strides),
                        strides: strides, share: isShared,
                        order: storageOrder)
        }
        set {
            var view = DenseTensor(shape: upper &- lower, storage: storage,
                                   offset: lower.index(stridedBy: strides),
                                   strides: strides, share: isShared,
                                   order: storageOrder)
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
        var result = self
        result._isShared = true
        return result
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
// DenseTensor initializer extensions
public extension DenseTensor {

    /// init(shape:element:order:
    @inlinable init(_ element: Element, _ shape: Shape, order: StorageOrder) {
        self.init(shape: shape, order: order)

        // initialize storage on the target device
        copy(from: RepeatedElement(shape, element: element), to: &self)
    }
    
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element == Element
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape: shape, order: order)
        _ = storage.hostBuffer.initialize(from: elements)
    }
    
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element: BinaryInteger, Element: Numeric
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape: shape, order: order)
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
        self.init(shape: shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
    
    /// init(shape:order:elements:
    /// implicitly casts from C.Element float -> Element integer
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
    {
        assert(shape.elementCount() == elements.count)
        self.init(shape: shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
}
