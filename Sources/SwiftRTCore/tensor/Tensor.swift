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
import Numerics


public let defaultElementName = "Element"
public let defaultTensorName = "Tensor"
public let defaultReferenceTensorName = "Reference"

//==============================================================================
/// Tensor
public struct Tensor<Shape, TensorElement>:
    TensorProtocol,
    MutableCollection,
    CustomStringConvertible,
    Logging
where Shape: TensorShape, TensorElement: StorageElement
{
    public typealias Index = ElementIndex<Shape>
    public typealias Element = TensorElement.Value
    
    /// the number of element
    public let count: Int
    /// `true` if the view will be shared by by multiple writers
    public var isShared: Bool
    /// element storage order in memory
    @noDerivative public let order: Order 
    /// a collection that maps logical coordinates to storage elements
    /// via the current storage order
    public var logicalElements: LogicalElements<Shape, TensorElement>
    /// the strides to traverse `shape` in logical coordinates
    public let logicalStrides: Shape
    /// the dimensions of the element space
    @noDerivative public let shape: Shape
    /// the element storage buffer.
    public var storage: StorageBufferType
    /// the logical storage buffer base index where this tensor's elements begin
    public let storageBase: Int
    /// The distance to the next element along each dimension
    public let strides: Shape
    /// the number of storage elements spanned by this tensor
    public let spanCount: Int

    //--------------------------------------------------------------------------
    // functional properties
    /// the unique storage id
    @inlinable public var id: Int { storage.id }
    /// the name of the collection
    @inlinable public var name: String {
        get { storage.name }
        set { storage.name = newValue }
    }

    /// `true` if the tensor elements are densely packed
    @inlinable public var isContiguous: Bool { spanCount == count }
    
    /// `true` if the tensor contains a single stored element. This is
    /// common for scalar tensors that are repeated.
    @inlinable public var isSingleElement: Bool { spanCount == 1 }

    /// `true` if the tensor value is zero
    @inlinable public var isZero: Bool { storage.isZero }

    //--------------------------------------------------------------------------
    /// init(
    /// Used to initialize an element collection subview
    @inlinable public init(
        shape: Shape,
        strides: Shape,
        count: Int,
        storage: StorageBufferType,
        storageBase: Int,
        spanCount: Int,
        order: Order,
        shared: Bool
    ) {
        // make sure the tensor view range is within the associated
        // storage buffer bounds.
        // Converts the logical last tensor element index to the corresponding
        // stored index and asserts it's less than buffer count
        assert(TensorElement.storedIndex(storageBase + spanCount - 1) <
                storage.countOf(type: TensorElement.Stored.self),
               "tensor storage range is out of bounds")

        // verify storage order is valid for rank
        assert((order != .NHWC || Shape.rank == 4) &&
               (order != .NDHWC || Shape.rank == 5))

        self.shape = shape
        self.strides = strides
        self.count = count
        self.storage = storage
        self.storageBase = storageBase
        self.spanCount = spanCount
        self.isShared = shared
        self.order = order
        logicalStrides = shape.strides(for: order)
        logicalElements = LogicalElements(count,
                                          shape,
                                          strides,
                                          storage,
                                          storageBase,
                                          order,
                                          spanCount)
    }

    //--------------------------------------------------------------------------
    /// init(value:shape:order:name
    /// Used to initialize a tensor with a single Element
    @inlinable public init(
        single value: TensorElement.Value,
        shape: Shape,
        order: Order,
        name: String
    ) {
        // verify storage order is valid for rank
        assert((order != .NHWC || Shape.rank == 4) &&
               (order != .NDHWC || Shape.rank == 5))

        self.shape = shape
        self.strides = Shape.zero
        self.storageBase = 0
        self.isShared = false
        self.count = shape.elementCount()
        self.spanCount = 1
        self.order = order
        let stored = TensorElement.stored(value: value)
        self.storage = StorageBufferType(storedElement: stored, name: name)
        logicalStrides = shape.strides(for: order)
        logicalElements = LogicalElements(count,
                                          shape,
                                          strides,
                                          storage,
                                          storageBase,
                                          order,
                                          spanCount)
    }

    //--------------------------------------------------------------------------
    /// init
    /// Used to represent a single zero value
    // Primarily used to minimize AD zero materialization problem
    @inlinable public init() {
        self.shape = Shape.one
        self.strides = Shape.one
        self.storageBase = 0
        self.isShared = false
        self.count = 1
        self.spanCount = 1
        self.order = .row
        self.storage = Context.zeroStorage
        logicalStrides = Shape.one
        logicalElements = LogicalElements(count,
                                          shape,
                                          strides,
                                          storage,
                                          storageBase,
                                          order,
                                          spanCount)
    }
}

//==============================================================================
/// Order
/// Specifies how to store multi-dimensional data in row-major (C-style)
/// or column-major (Fortran-style) order in memory.
/// These names are following the numpy naming convention
public enum Order: Int, Codable {
    /// Data is ordered in column-major dense sequential format.
    /// The leading dimension is the stride (in elements) to the beginning
    /// of next column in memory.
    case col

    /// Data is ordered in row-major dense sequential format.
    /// The second dimension is the stride (in elements) to the beginning
    /// of next row in memory.
    case row
    
    /// Data is ordered in column-major ordered tiles of 32 columns.
    /// The leading dimension is the stride (in elements) to the beginning
    /// of next group of 32-columns. For example, if the matrix has 33 columns
    /// and 2 rows, then the leading dimension must be at least (32) * 2 = 64.
    case colTiled32

    //--------------------------------------------------------------------------
    /// Data is ordered as batch N of (rows H, columns W, channels C)
    /// This order is only valid for 4D tensors. 
    case NHWC

    /// Data is ordered as batch N of (depths D, rows H, columns W, channels C)
    /// This order is only valid for 5D tensors. 
    case NDHWC

    //--------------------------------------------------------------------------
    // NVIDIA native tensor core formats 

    /// Data is ordered in column-major ordered tiles of composite tiles
    /// with total 32 columns and 8 rows. A tile is composed of interleaved
    /// inner tiles of 4 columns within 4 even or odd rows in an alternating
    /// pattern. The leading dimension is the stride (in elements) to the
    /// beginning of the first 32 column x 8 row tile for the next 32-wide
    /// group of columns. For example, if the matrix has 33 columns and
    /// 1 row, the leading dimension must be at least (32 * 8) * 1 = 256.
    /// NOTE: this order is needed for the B matrix on NVIDIA Turing
    /// Architecture GPUs, i.e. SM version = 72 and 75, for maximum tensor
    /// core integer GEMM performance.
    // ORDER_COL4_4R2_8C
    case colTiledTC32x8

    /// Data is ordered in column-major ordered tiles of composite tiles
    /// with total 32 columns ands 32 rows. Element offset within the tile
    /// is calculated as
    ///     index = (((row%8) / 2 * 4 + row / 8) * 2 + row % 2) * 32 + col
    /// Leading dimension is the stride (in elements) to the beginning
    /// of the first 32 column x 32 row tile for the next 32-wide group
    /// of columns. E.g. if matrix has 33 columns and 1 row, ld must be
    /// at least (32*32)*1 = 1024.
    /// NOTE: this order is needed for the B matrix on NVIDIA Ampere
    /// Architecture GPUs, i.e. SM version >= 80, for maximum tensor
    /// core integer GEMM performance.
    // ORDER_COL32_2R_4R4
    case colTiledTC32x32

    // aliases
    public static let C = row, F = col, A = -1
    public static var defaultOrder: Order = Order.row
}

@usableFromInline let _messageLayoutsMustMatch = "input Order must match"

@usableFromInline func layoutsMatch(_ layouts: Order...) -> Bool {
    layouts.first(where: { $0 != layouts[0] }) == nil
}

//==============================================================================
/// DifferentiableTensor
///
/// While these protocols are not strictly necessary, they are used
/// to reduce the number of generic requirements when writing
/// `@differentiable` attributes
///
public protocol TensorProtocol: Logging {
    associatedtype Shape: TensorShape
    associatedtype TensorElement: StorageElement
}

public protocol DifferentiableTensor: TensorProtocol & Differentiable
where Self == TangentVector, TensorElement.Value: DifferentiableNumeric {}

/// DifferentiableNumeric
public protocol DifferentiableNumeric:
    Differentiable & Numeric where Self == TangentVector {}

extension Float: DifferentiableNumeric {}
extension Double: DifferentiableNumeric {}

extension Complex: DifferentiableNumeric
where RealType: Differentiable, RealType.TangentVector == RealType {}

// Differentiable conformance
extension Tensor: Differentiable & DifferentiableTensor
    where Element: DifferentiableNumeric
{
    public typealias TangentVector = Self
}

extension Tensor: AdditiveArithmetic where Element: Numeric {
    @inlinable public static var zero: Self { Tensor() }
    @inlinable public static var one: Self { Tensor(1, name: "One") }
}

//==============================================================================
// Tensor Codable
@usableFromInline enum TensorCodingKeys: String, CodingKey {
    case data, shape, name, order
}

extension Tensor: Codable where Element: Codable {
    /// encodes the contents of the array
    @inlinable public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: TensorCodingKeys.self)
        try container.encode(storage.name, forKey: .name)
        try container.encode(shape, forKey: .shape)
        try container.encode(order, forKey: .order)
        var dataContainer = container.nestedUnkeyedContainer(forKey: .data)
        if isBufferIterable {
            try self.buffer.forEach {
                try dataContainer.encode($0)
            }
        } else {
            try self.forEach {
                try dataContainer.encode($0)
            }
        }
    }
    
    @inlinable public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TensorCodingKeys.self)
        let name = try container.decode(String.self, forKey: .name)
        let shape = try container.decode(Shape.self, forKey: .shape)
        let order = try container.decode(Order.self, forKey: .order)
        var dataContainer = try container.nestedUnkeyedContainer(forKey: .data)
        self = Self(shape: shape, order: order)
        self.name = name

        assert(self.count == dataContainer.count)
        var buffer = self.mutableBuffer
        for i in buffer.indices {
            buffer[i] = try dataContainer.decode(Element.self)
        }
    }
}

//==============================================================================
/// ElementIndex
/// Common index type used to iterate through collection elements
/// `position` is the index position in n-dimensional space
/// `sequencePosition` is the linear sequence position when iterating
/// and used for comparison
public struct ElementIndex<Shape>: Comparable, Codable
    where Shape: TensorShape
{
    /// the logical position along each axis
    public let position: Shape
    /// linear sequence position
    public let sequencePosition: Int

    // init(position:sequencePosition:
    @inlinable public init(_ position: Shape, _ sequencePosition: Int) {
        self.position = position
        self.sequencePosition = sequencePosition
    }

    /// init(sequencePosition:
    /// initializer for collections that ignore logical position
    @inlinable public init(at sequencePosition: Int) {
        self.position = Shape.zero
        self.sequencePosition = sequencePosition
    }

    /// incremented(lower:upper:
    /// increments `position` with the range `lower..<upper`
    @inlinable
    public func incremented(between lower: Self, and upper: Self) -> Self {
        let pos = position.incremented(between: lower.position,
                                       and: upper.position)
        return ElementIndex(pos, sequencePosition + 1)
    }
    
    @inlinable public func linearIndex(_ strides: Shape) -> Int {
        position.index(stridedBy: strides)
    }

    // Equatable
    @inlinable public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.sequencePosition == rhs.sequencePosition
    }
    
    // Comparable
    @inlinable public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.sequencePosition < rhs.sequencePosition
    }
}

//==============================================================================
// Tensor collection and sub view extensions
public extension Tensor {
    @inlinable var isBufferIterable: Bool {
        isSingleElement || isContiguous
    }
    
    //--------------------------------------------------------------------------
    // sequential buffer element iterators
    @inlinable var buffer: BufferElements<Shape,TensorElement> {
        BufferElements(tensor: self)
    }
    
    @inlinable var mutableBuffer: BufferElements<Shape,TensorElement> {
        mutating get { BufferElements(tensor: &self) }
    }
    
    //--------------------------------------------------------------------------
    // logical coordinate element iterators
    @inlinable var elements: LogicalElements<Shape,TensorElement> {
        logicalElements.prepareForRead()
        return logicalElements
    }
    
    @inlinable var mutableElements: LogicalElements<Shape,TensorElement> {
        mutating get {
            logicalElements.prepareForReadWrite()
            return logicalElements
        }
    }

    //--------------------------------------------------------------------------
    /// the starting index zero relative to the storage buffer
    @inlinable var startIndex: Index {
        logicalElements.startIndex
    }
    
    //--------------------------------------------------------------------------
    /// the ending index zero relative to the storage buffer
    @inlinable var endIndex: Index {
        logicalElements.endIndex
    }

    //--------------------------------------------------------------------------
    /// makeIndex(position:
    /// makes an index from a logical position within `shape`
    /// - Parameters:
    ///  - position: the n-dimensional coordinate position within `shape`
    /// - Returns: the index
    @inlinable func makeIndex(at position: Shape) -> Index {
        Index(position, position.index(stridedBy: logicalStrides))
    }

    //--------------------------------------------------------------------------
    /// index(i:
    @inlinable func index(after i: Index) -> Index {
        logicalElements.index(after: i)
    }

    //--------------------------------------------------------------------------
    // elemment subscript
    @inlinable subscript(i: Index) -> Element {
        get {
            usingAppThreadQueue {
                logicalElements.prepareForRead()
                return logicalElements[i]
            }
        }
        set {
            usingAppThreadQueue {
                prepareForWrite(using: Context.currentQueue)
                logicalElements.prepareForReadWrite()
                logicalElements[i] = newValue
            }
        }
    }

    //--------------------------------------------------------------------------
    // sub view subscript
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable subscript(lower: Shape, upper: Shape) -> Self {
        get { createView(lower, upper, isShared) }
        set {
            prepareForWrite(using: Context.currentQueue)
            var view = createView(lower, upper, true)
            copy(from: newValue, to: &view)
        }
    }
    
    //--------------------------------------------------------------------------
    // creates a tensor subview
    @inlinable func createView(
        _ lower: Shape,
        _ upper: Shape,
        _ share: Bool
    ) -> Self {
        let shape = upper &- lower
        let count = shape.elementCount()
        let spanCount = strides.areSequential(for: shape) ? count :
                shape.spanCount(stridedBy: strides)

        return Tensor(
            shape: shape,
            strides: strides,
            count: count,
            storage: storage,
            storageBase: storageBase + lower.index(stridedBy: strides),
            spanCount: spanCount,
            order: order,
            shared: share)
    }

    //--------------------------------------------------------------------------
    /// `prepareForWrite`
    /// called before a write operation to ensure that the storage buffer
    /// is unique for this tensor unless it `isShared`
    /// It also expands repeated tensors to a full dense storage
    /// representation for write, which most often happens via element
    /// subscripting.
    @inlinable mutating func prepareForWrite(using queue: PlatformType.Device.Queue) {
        // if repeated then expand to full dense tensor
        if spanCount < count {
            var expanded = Tensor(like: self)

            diagnostic(.expanding, "\(name)(\(id)) " +
                    "\(Element.self)[\(spanCount)] to: \(expanded.name)"
                    + "(\(expanded.id)) \(Element.self)[\(expanded.count)]",
                categories: [.dataCopy, .dataExpanding])

            // do an indexed copy
            copy(from: self, to: &expanded)
            self = expanded

        } else if !(isKnownUniquelyReferenced(&storage) || isShared) {
            // if not uniquely held then copy before creating the shared view
            diagnostic(.mutation, "\(storage.name)(\(storage.id)) " +
                        "\(Element.self)[\(count)]",
                       categories: [.dataCopy, .dataMutation])
            
            storage = StorageBufferType(type: Element.self, copying: storage,
                                        using: queue)
            logicalElements = LogicalElements(tensor: self)
        }
    }

    //--------------------------------------------------------------------------
    /// - Returns: the collection elements as a 1D Swift array
    @inlinable var flatArray: [Element] {
        usingAppThreadQueue {
            isBufferIterable ? [Element](buffer) : [Element](elements)
        }
    }
}

//==============================================================================
/// Derivative registration
extension Tensor where TensorElement.Value: DifferentiableNumeric {
    // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
    
    @derivative(of: subscript)
    @usableFromInline func _vjpSubscript(lower: Shape, upper: Shape)
        -> (value: Self, pullback: (Self) -> Self)
    {
        return (self[lower, upper], { v in
            var result = zeros(like: self)
            result[lower, upper] = v
            return result
        })
    }
}

//==============================================================================
// Tensor read write access
public extension Tensor {
    //--------------------------------------------------------------------------
    /// `read`
    /// Synchronizes the collection of stored elements with the caller
    /// for reading. This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `Collection`
    /// enumeration via `indices` or integer subscripting.
    @inlinable func read() -> UnsafeBufferPointer<TensorElement.Stored> {
        read(using: Context.appThreadQueue)
    }
    
    //--------------------------------------------------------------------------
    /// `read(queue:
    /// Synchronizes the collection of elements for reading
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable func read(
        using queue: DeviceQueue
    ) -> UnsafeBufferPointer<TensorElement.Stored> {
        let (i, storedCount) = TensorElement
                .storedRange(start: storageBase, count: spanCount)

        return storage.read(type: TensorElement.Stored.self,
                            at: i, count: storedCount, using: queue)
    }

    //--------------------------------------------------------------------------
    /// `deviceRead(queue:
    /// Synchronizes the collection of elements for reading
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable func deviceRead(using queue: DeviceQueue) -> UnsafeRawPointer {
        UnsafeRawPointer(read(using: queue).baseAddress!)
    }
    
    //--------------------------------------------------------------------------
    /// `readWrite`
    /// Synchronizes the collection of elements with the caller for read write
    /// This function blocks until the elements are available.
    /// `Elements` are accessed by the application using `MutableCollection`
    /// enumeration via `indices` or subscripting.
    @inlinable mutating func readWrite()
        -> UnsafeMutableBufferPointer<TensorElement.Stored> {
        readWrite(using: Context.appThreadQueue)
    }
    
    //--------------------------------------------------------------------------
    /// `readWrite(queue:`
    /// Synchronizes the collection of elements with the caller for read write
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable mutating func readWrite(using queue: PlatformType.Device.Queue)
    -> UnsafeMutableBufferPointer<TensorElement.Stored>
    {
        prepareForWrite(using: queue)

        let (i, storedCount) = TensorElement
                .storedRange(start: storageBase, count: spanCount)
        
        return storage.readWrite(type: TensorElement.Stored.self,
                                 at: i, count: storedCount, using: queue)
    }
    
    //--------------------------------------------------------------------------
    /// `deviceReadWrite(queue:`
    /// Synchronizes the collection of elements with the caller for read write
    /// using the specified `queue`. This function is non blocking, and
    /// the elements will be available when the request reaches the
    /// head of the queue.
    ///
    /// - Parameter queue: the device queue to use for synchronization
    @inlinable mutating func deviceReadWrite(
        using queue: PlatformType.Device.Queue
    ) -> UnsafeMutableRawPointer {
        UnsafeMutableRawPointer(readWrite(using: queue).baseAddress!)
    }
}

//==============================================================================
// Tensor element properties
public extension Tensor {
    /// first
    /// - Returns: the first element in the tensor
    @inlinable var first: Element {
        TensorElement.getValue(from: read(), at: storageBase)
    }

    /// element
    /// can get and set the value of a single element tensor.
    /// - Returns: the only element in the tensor
    @differentiable(where TensorElement.Value: DifferentiableNumeric)
    @inlinable var element: Element {
        get {
            assert(count == 1, "the `element` property expects " +
                "the tensor to have a single Element. Use `first` for sets")
            return TensorElement.getValue(from: read(), at: storageBase)
        }
        set {
            assert(count == 1, "the `element` property expects " +
                "the tensor to have a single Element")
            TensorElement.set(value: newValue, in: readWrite(), at: storageBase)
        }
    }

    @derivative(of: element)
    @inlinable func vjpElement() -> (
      value: Element,
      pullback: (Element) -> Self
    ) where Element: DifferentiableNumeric {
      (element, { v in
        var result = zeros(like: self)
        result.element = v
        return result
      })
    }
}

