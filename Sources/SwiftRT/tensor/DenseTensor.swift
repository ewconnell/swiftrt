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
    /// `true` if the view will be shared by by multiple writers
    public let isShared: Bool
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
    /// init(shape:
    @inlinable public init(
        from lower: Shape,
        to upper: Shape,
        storage: StorageBufferType<Element>? = nil,
        strides: Shape? = nil,
        share: Bool = false,
        order: StorageOrder = .rowMajor,
        element value: Element? = nil
    ) {
        assert(storage == nil || lower == Shape.zero,
               "The lower bound of new storage must be zero")
        self.shape = upper &- lower
        var sequentialStrides = shape.sequentialStrides()
        let elementCount = shape.elementCount()
        self.elementCount = elementCount
        self.storageOrder = order
        self.isShared = share
        self.storage = storage ?? StorageBufferType(count: elementCount,
                                                    name: Self.name,
                                                    element: value)
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
        if isSequential {
            getElement = { [storage = self.storage] in
                storage.hostBuffer[$0.sequencePosition]
            }
            setElement = { [storage = self.storage] in
                storage.hostBuffer[$0.sequencePosition] = $1
            }
        } else {
            getElement = { [storage = self.storage, strides = self.strides] in
                storage.hostBuffer[$0.linearIndex(with: strides)]
            }
            setElement = { [storage = self.storage, strides = self.strides] in
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
                        order: storageOrder, element: nil)
        }
        set {
            var view = DenseTensor(from: lower, to: upper, storage: storage,
                                   strides: strides, share: isShared,
                                   order: storageOrder, element: nil)
            copy(from: newValue, to: &view)
        }
    }
    
    //--------------------------------------------------------------------------
    /// share(lower:upper:
    @inlinable
    public mutating func share(from lower: Shape, to upper: Shape) -> Self {
        // if not uniquely held then copy before creating the shared view
        if !isKnownUniquelyReferenced(&storage) {
            diagnostic("\(mutationString) \(storage.name)(\(storage.id)) " +
                "\(Element.self)[\(elementCount)]",
                categories: [.dataCopy, .dataMutation])

            storage = StorageBufferType(copying: storage)
        }

        // return shared view
        return DenseTensor(from: lower, to: upper, storage: storage,
                           strides: strides, share: true,
                           order: storageOrder, element: nil)
    }
}

