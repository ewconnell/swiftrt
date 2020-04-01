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
public struct DenseTensor<Shape, Element>: MutableTensor, MutableCollection
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
    /// the linear element offset where the view begins
    public let bufferOffset: Int
    /// `true` if elements are in row major contiguous order
    public let isSequential: Bool
    /// Specifies whether data is stored in row-major (C-style)
    /// or column-major (Fortran-style) order in memory.
    public let storageOrder: StorageOrder
    /// the dimensions of the element space
    public let shape: Shape
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
    @inlinable public var startIndex: Index { Index(Shape.zero, 0) }
    /// the ending index zero relative to the storage buffer
    @inlinable public var endIndex: Index { Index(shape, elementCount) }

    //-----------------------------------
    /// a function defined during initialization to get storage elements
    @usableFromInline let getElement: (Index) -> Element
    /// a function defined during initialization to set storage elements
    @usableFromInline let setElement: (Index, Element) -> Void

    
    //--------------------------------------------------------------------------
    /// init(lower:upper:storage:strides:share:order
    @inlinable public init(
        from lower: Shape,
        to upper: Shape,
        storage: StorageBufferType<Element>? = nil,
        strides: Shape? = nil,
        share: Bool = false,
        order: StorageOrder = .rowMajor
    ) {
        assert(storage == nil || lower == Shape.zero,
               "The lower bound of new storage must be zero")
        self.shape = upper &- lower
        var sequentialStrides = shape.sequentialStrides()
        let count = shape.elementCount()
        self.elementCount = count
        self.storageOrder = order
        self._isShared = share
        self.storage = storage ?? StorageBufferType(count: count, name: Self.name)
        
        if let strides = strides {
            self.strides = strides
            self.spanCount = shape.spanCount(with: strides)
            sequentialStrides[0] = strides[0]
            self.isSequential = strides == sequentialStrides
        } else {
            self.isSequential = true
            self.strides = sequentialStrides
            self.spanCount = elementCount
        }
        self.bufferOffset = lower.linearIndex(with: self.strides)

        //----------------------------------
        // element access functions depending on memory order
        // shadow these variables for implicit capture
        let storage = self.storage
        let strides = self.strides

        if isSequential {
            getElement = {
                storage.hostBuffer[$0.sequencePosition]
            }
            setElement = {
                storage.hostBuffer[$0.sequencePosition] = $1
            }
        } else {
            getElement = {
                storage.hostBuffer[$0.linearIndex(with: strides)]
            }
            setElement = {
                storage.hostBuffer[$0.linearIndex(with: strides)] = $1
            }
        }
    }
    
    //--------------------------------------------------------------------------
    // Collection
    /// index(i:
    @inlinable public func index(after i: Index) -> Index {
        i.incremented(between: startIndex, and: endIndex)
    }
    
    // elemment subscript
    @inlinable public subscript(index: Index) -> Element {
        get { getElement(index) }
        set { setElement(index, newValue) }
    }
    
    //--------------------------------------------------------------------------
    // view subscripts
    @inlinable public subscript(lower: Shape, upper: Shape) -> Self {
        get {
            DenseTensor(from: lower, to: upper, storage: storage,
                        strides: strides, share: isShared,
                        order: storageOrder)
        }
        set {
            var view = DenseTensor(from: lower, to: upper, storage: storage,
                                   strides: strides, share: isShared,
                                   order: storageOrder)
            copy(from: newValue, to: &view)
        }
    }
    
    //--------------------------------------------------------------------------
    /// shared(
    @inlinable public mutating func shared() -> Self {
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
// DenseTensor initializer extensions
public extension DenseTensor {

    /// init(shape:order:
    @inlinable init(_ shape: Shape, order: StorageOrder = .C) {
        self.init(from: Shape.zero, to: shape, order: order)
    }
    
    /// init(shape:element:order:
    @inlinable init(_ element: Element, _ shape: Shape, order: StorageOrder) {
        self.init(from: Shape.zero, to: shape, order: order)

        // initialize storage on the target device
        copy(from: RepeatedElement(shape, element: element), to: &self)
    }
    
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element == Element
    {
        assert(shape.elementCount() == elements.count)
        self.init(from: Shape.zero, to: shape, order: order)
        _ = storage.hostBuffer.initialize(from: elements)
    }
    
    /// init(shape:elements:order:
    /// implicitly casts from C.Element integer -> Element
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element: BinaryInteger, Element: Numeric
    {
        assert(shape.elementCount() == elements.count)
        self.init(from: Shape.zero, to: shape, order: order)
        let lazyElements = elements.lazy.map { Element(exactly: $0)! }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
    
    /// init(shape:order:elements:
    /// implicitly casts from C.Element float -> Element integer
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryInteger
    {
        assert(shape.elementCount() == elements.count)
        self.init(from: Shape.zero, to: shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
    
    /// init(shape:order:elements:
    /// implicitly casts from C.Element float -> Element integer
    @inlinable init<C>(_ elements: C, _ shape: Shape, order: StorageOrder = .C)
        where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
    {
        assert(shape.elementCount() == elements.count)
        self.init(from: Shape.zero, to: shape, order: order)
        let lazyElements = elements.lazy.map { Element($0) }
        _ = storage.hostBuffer.initialize(from: lazyElements)
    }
}