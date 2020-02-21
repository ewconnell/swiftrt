//******************************************************************************
// Copyright 2019 Google LLC
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
/// TensorView protocol
/// A TensorView object is the primary interface for working with data.
/// Specialized shaped instances such as Vector, Matrix, Volume, etc..
/// conform to this protocol
///
public protocol TensorView: Logging {
    /// the type of element stored by the tensor
    associatedtype Element
    /// tensor shape
    associatedtype Shape: ShapeProtocol
    /// A concrete type used in generics to pass Boolean values
    associatedtype BoolView: TensorView where
        BoolView.Element == Bool, BoolView.Shape == Shape
    /// A concrete type used in generics to return index results
    associatedtype IndexView: TensorView where
        IndexView.Element == IndexType, IndexView.Shape == Shape

    //--------------------------------------------------------------------------
    // properties
    /// a label for the type used as a default name in diagnostics
    static var diagnosticName: String { get }
    /// class reference to the underlying byte buffer
    var elementBuffer: BufferId { get }
    /// if `true` then readWrite buffer access will not cause copy-on-write
    var isMutable: Bool { get }
    /// the shape of the view used for indexing
    var shape: Shape { get }
    /// the linear element offset where the view begins
    var offset: Int { get }
    
    //--------------------------------------------------------------------------
    /// fully specified used for creating tensors
    init(shape: Shape, elementBuffer: BufferId, offset: Int, isMutable: Bool)

    //--------------------------------------------------------------------------
    /// creates a new dense tensor of the same type with the specified extents
    func createDense(with extents: Shape.Array, name: String?) -> Self
    /// creates a new dense tensor where `Element` equals `Bool`
    /// with the specified extents
    func createBoolTensor(with extents: Shape.Array) -> BoolView
    /// creates a new dense tensor where `Element` equals `IndexType`
    /// with the specified extents and initial values
    func createIndexTensor(with extents: Shape.Array) -> IndexView
}

//==============================================================================
//
public extension TensorView {
    @inlinable
    var elements: ElementBuffer<Element, Shape> {
        fatalError()
    }

    @inlinable
    var mutableElements: MutableElementBuffer<Element, Shape> {
        fatalError()
    }
}

//==============================================================================
/// ScalarType
/// Used primarily for serialization, C APIs, and Cuda kernels
// TODO: maybe remove this after Cuda integration if not used
public enum ScalarType: Int {
    // integers
    case real8U, real8I, real16U, real16I, real32U, real32I, real64U, real64I
    // floats
    case real16F, real32F, real64F
    // non numeric
    case bool
}

//==============================================================================
// TensorView default implementation
public extension TensorView {
    /// first
    /// - Returns: the first element in the tensor
    @inlinable
    var first: Element {
        readOnly()[0]
    }

    /// element
    /// can get and set the value of a single element tensor.
    /// - Returns: the only element in the tensor
    @inlinable
    @_semantics("autodiff.nonvarying")
    var element: Element {
        get {
            assert(shape.isScalar, "the `element` property expects " +
                "the tensor to have a single Element. Use `first` for sets")
            return first
        }
        set {
            assert(shape.isScalar, "the `element` property expects " +
                "the tensor to have a single Element")
            readWrite()[0] = newValue
        }
    }
}

//==============================================================================
// TensorView default implementation
public extension TensorView {
    //--------------------------------------------------------------------------
    /// the number of elements in the collection
    @inlinable
    var count: Int { shape.count }
    /// the extents of the view
    @inlinable
    @_semantics("autodiff.nonvarying")
    var extents: Shape.Array { shape.extents }
    /// `true` if the values are contiguosly arranged in memory
    @inlinable
    var isContiguous: Bool { shape.isContiguous }
    /// the number of items in the tensor, which is equal to `extents[0]`
    @inlinable
    var items: Int { shape.items }
    /// the name of the view, which can optionally be set to aid in debugging
    @inlinable
    var name: String { tensorArray.name }
    /// the number of dimensions in the view
    @inlinable
    var rank: Int { shape.rank }
    /// the strides of the tensor elements
    @inlinable
    var strides: Shape.Array { shape.strides }
    /// an array of viewed elements
    @inlinable
    var flatArray: [Element] { [Element](elements()) }
    /// repeated(to extents:
    @inlinable
    func repeated(to extents: Shape.Array) -> Self {
        return Self(shape: shape.repeated(to: extents),
                    tensorArray: tensorArray,
                    viewOffset: viewOffset,
                    isMutable: isMutable)
    }
    
    @inlinable
    func repeated(to extents: Shape.Tuple) -> Self {
        repeated(to: Shape.Array(extents))
    }
}

//==============================================================================
// TensorView view creation functions
public extension TensorView {
    //--------------------------------------------------------------------------
    /// makePositive(index:
    @inlinable
    @_semantics("autodiff.nonvarying")
    func makePositive(index: Shape.Tuple) -> Shape.Array {
        var result = Shape.Array(index)
        for i in 0..<result.count {
            if result[i] < 0 { result[i] += extents[i] }
        }
        return result
    }
    
    //--------------------------------------------------------------------------
    /// view
    /// Creates an immutable subview
    @inlinable
    func view(at index: Shape.Tuple, extents: Shape.Tuple,
              strides: Shape.Tuple? = nil) -> Self
    {
        view(at: Shape.Array(index),
             extents: Shape.Array(extents),
             strides: Shape.Array(strides))
    }
    
    @inlinable
    func view(at index: Shape.Array, extents: Shape.Array,
              strides: Shape.Array? = nil) -> Self
    {
        createView(at: index, extents: extents,
                   strides: strides ?? self.strides,
                   isMutable: isMutable)
    }
    
    //--------------------------------------------------------------------------
    /// mutableView
    /// A mutableView does not perform a copy-on-write check for
    /// a readWrite buffer access. However, the data is copied the first time
    /// if the tensor is not uniquely held. Mutable views derived from a
    /// mutable view will not copy the data irrespective to reference count.
    /// This allows for multi-threaded tensor write operations.
    @inlinable
    mutating func mutableView(at index: Shape.Tuple, extents: Shape.Tuple,
                              strides: Shape.Tuple? = nil) -> Self
    {
        mutableView(at: Shape.Array(index),
                    extents: Shape.Array(extents),
                    strides: Shape.Array(strides))
    }
    
    @inlinable
    mutating func mutableView(at index: Shape.Array, extents: Shape.Array,
                              strides: Shape.Array? = nil) -> Self
    {
        // copy the tensor array if not uniquely held or
        // if this is a broadcasted value
        copyIfMutates(using: Platform.service.currentQueue)
        
        // return a mutable view against a safe dense tensor array
        return createView(at: index, extents: extents,
                          strides: strides ?? self.strides,
                          isMutable: true)
    }

    @inlinable
    mutating func mutableView() -> Self {
        mutableView(at: Shape.zeros, extents: self.extents)
    }

    //--------------------------------------------------------------------------
    /// createView
    /// Returns a view of the tensorArray relative to this view
    @usableFromInline
    internal func createView(at index: Shape.Array, extents: Shape.Array,
                             strides: Shape.Array, isMutable: Bool) -> Self
    {
        // validate
        assert(index.count == shape.rank && extents.count == shape.rank)
        assert(shape.contains(index: index, extents: extents))
        
        // the subview offset is the current plus the offset of index
        let subViewOffset = viewOffset + shape.linearIndex(of: index)
        return Self(shape: Shape(extents: extents, strides: strides),
                    tensorArray: tensorArray,
                    viewOffset: subViewOffset,
                    isMutable: isMutable)
    }
    
    //--------------------------------------------------------------------------
    /// transposed
    /// transposes indexing axes of the tensor
    /// - Parameter with: and optional axes permutation order. If `nil` the
    /// last two dimensions are swapped.
    @inlinable
    func transposed(with permutations: Shape.Tuple? = nil) -> Self {
        guard self.rank > 1 else { return self }
        let permuted = self.shape.transposed(with: Shape.Array(permutations))
        return Self(shape: permuted,
                    tensorArray: tensorArray,
                    viewOffset: viewOffset,
                    isMutable: isMutable)
    }
}

//==============================================================================
// TensorView buffer access functions
public extension TensorView {
    //--------------------------------------------------------------------------
    /// isUniqueReference
    /// `true` if this view is the only view holding a reference to tensorArray
    @inlinable
    mutating func isUniqueReference() -> Bool {
        isKnownUniquelyReferenced(&tensorArray)
    }
    
    //--------------------------------------------------------------------------
    /// writeWillMutateView
    /// `true` if write access will cause the underlying `tensorArray`
    ///  to be copied
    @inlinable
    mutating func writeWillMutateView() -> Bool {
        !isUniqueReference() && !isMutable
    }
    
    //--------------------------------------------------------------------------
    /// copyIfMutates
    /// Creates a copy of the tensorArray if read-write access causes mutation
    /// - Parameter using: the device queue to use for data transfer
    /// - Returns: `true` if the `tensorArray` was copied
    @inlinable
    mutating func copyIfMutates(using queue: DeviceQueue) {
        guard writeWillMutateView() else { return }
        
        // the reference is not unique so a copy of the array must be made
        diagnostic("\(mutationString) \(name)(\(tensorArray.trackingId)) " +
            "\(String(describing: Element.self))[\(shape.count)]",
            categories: [.dataCopy, .dataMutation])

        // create the new array and do a simple copy of the elements
        tensorArray = TensorArray<Element>(copying: tensorArray, using: queue)
    }
    
    //--------------------------------------------------------------------------
    /// synchronizeQueues
    /// If the queue is changing, then this creates an event and
    /// records it onto the end of the lastQueue, then records a wait
    /// on the new queue. This insures the lastQueue finishes before
    /// the new one begins
    @inlinable
    func synchronize(queue lastQueue: DeviceQueue?, with nextQueue: DeviceQueue)
    {
        if let lastQueue = lastQueue, nextQueue.id != lastQueue.id {
            let event = lastQueue.createEvent()
            diagnostic(
                "\(nextQueue.deviceName)_\(nextQueue.name) will wait for " +
                    "\(lastQueue.deviceName)_\(lastQueue.name) " +
                "using QueueEvent(\(event.id))",
                categories: .queueSync)
            nextQueue.wait(for: lastQueue.record(event: event))
        }
    }
    
    //--------------------------------------------------------------------------
    /// readOnly(using queue:
    /// Returns a read only device memory buffer synced with the specified
    /// queue.
    @inlinable
    func readOnly(using queue: DeviceQueue? = nil)
        -> UnsafeBufferPointer<Element>
    {
        // if no queue is specified then use the hostQueue
        let deviceQueue = queue ?? Platform.service.applicationQueue
        
        // sync queues
        synchronize(queue: tensorArray.lastMutatingQueue, with: deviceQueue)
        
        // get the buffer
        let buffer = tensorArray.readOnly(using: deviceQueue)
        
        // if `queue` is nil then the deviceQueue is the hostQueue
        // and the caller wants to synchronize with the app thread
        if queue == nil {
            assert(deviceQueue.memoryAddressing == .unified)
            deviceQueue.waitUntilQueueIsComplete()
        }
        
        return UnsafeBufferPointer(
            start: buffer.baseAddress!.advanced(by: viewOffset),
            count: shape.spanCount)
    }
    
    //--------------------------------------------------------------------------
    /// deviceReadOnly(using queue:
    /// Returns a read only device raw memory pointer synced with the specified
    /// queue.
    @inlinable
    func deviceReadOnly(using queue: DeviceQueue? = nil)
        -> UnsafeRawPointer
    {
        UnsafeRawPointer(readOnly(using: queue).baseAddress!)
    }
    
    //--------------------------------------------------------------------------
    /// readWrite(using queue:
    /// Returns a read write device memory buffer synced with the specified
    /// queue.
    @inlinable
    mutating func readWrite(using queue: DeviceQueue? = nil)
        -> UnsafeMutableBufferPointer<Element>
    {
        precondition(!tensorArray.isReadOnly, "the tensor is read only")
        let deviceQueue = queue ?? Platform.service.applicationQueue
        
        // sync queues
        synchronize(queue: tensorArray.lastMutatingQueue,
                        with: deviceQueue)
        // mutating write?
        copyIfMutates(using: deviceQueue)
        
        // get the buffer
        let buffer = tensorArray.readWrite(using: deviceQueue)
        
        // if `queue` is nil then the deviceQueue is the hostQueue
        // and the caller wants to synchronize with the app thread
        if queue == nil {
            assert(deviceQueue.memoryAddressing == .unified)
            deviceQueue.waitUntilQueueIsComplete()
        }
        
        return UnsafeMutableBufferPointer(
            start: buffer.baseAddress!.advanced(by: viewOffset),
            count: shape.spanCount)
    }
    
    //--------------------------------------------------------------------------
    /// deviceReadWrite(using queue:
    /// Returns a read write device raw memory pointer synced with the specified
    /// queue.
    @inlinable
    mutating func deviceReadWrite(using queue: DeviceQueue? = nil) throws
        -> UnsafeMutableRawPointer
    {
        UnsafeMutableRawPointer(readWrite(using: queue).baseAddress!)
    }
}

//==============================================================================
public extension TensorView {
    //--------------------------------------------------------------------------
    /// hostMultiWrite
    /// divides a tensor into mutable batches and concurrently passes them
    /// to the `body` for processing
    /// - Parameter batchSize: the number of items to process at a time. The
    /// default is the total divided by the number of active cores
    /// - Parameter synchronous: if `true` the batches will be executed
    /// synchronously to aid in debugging
    /// - Parameter body: the function to perform
    mutating func hostMultiWrite(
        batchSize: Int? = nil,
        synchronous: Bool = false,
        _ body: @escaping (_ view: inout Self) throws -> Void) throws
    {
        assert(batchSize == nil || batchSize! <= extents[0])
        let queue = Platform.service.applicationQueue
        var fullView = self.mutableView()
        let group = DispatchGroup()
        let batchQueue = DispatchQueue(label: "hostMultiWrite",
                                       attributes: .concurrent)
        let batchSize = batchSize ?? {
            let size = Int(extents[0]) /
                ProcessInfo.processInfo.activeProcessorCount
            return size == 0 ? Int(extents[0]) : size
        }()
        let remainder = Int(extents[0]) % batchSize
        
        // do the work
        func queueBatch(item: Int, count: Int) throws {
            var batchView = fullView[item..|count]
            if synchronous {
                try body(&batchView)
            } else {
                batchQueue.async(group: group) {
                    do {
                        try body(&batchView)
                    } catch {
                        Platform.service.writeLog("\(error)")
                    }
                }
            }
        }
        
        // ensure the data is local
        _ = fullView.readWrite(using: queue)
        
        // launch the batches
        let lastBatchIndex = Int(extents[0]) - remainder
        for i in stride(from: 0, to: lastBatchIndex, by: batchSize) {
            try queueBatch(item: i, count: batchSize)
        }
        
        // process remaining items
        if remainder > 0 {
            try queueBatch(item: lastBatchIndex, count: remainder)
        }
        group.wait()
    }
}

//==============================================================================
// Codable
public enum TensorCodingKeys: String, CodingKey { case extents, data }

public extension TensorView {
    /// encodes the contents of the array
    @inlinable
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: TensorCodingKeys.self)
        try container.encode(extents, forKey: .extents)
        try container.encode(tensorArray, forKey: .data)
    }
    
    @inlinable
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: TensorCodingKeys.self)
        let extents = try container.decode(Shape.Array.self, forKey: .extents)
        let array = try container.decode(TensorArray<Element>.self,
                                         forKey: .data)
        self = Self(shape: Shape(extents: extents), tensorArray: array,
                    viewOffset: 0, isMutable: false)
    }
}

//==============================================================================
// == operator to simplify unit test syntax
public extension TensorView {
    /// compares the flat elements of self with a Swift array of elements
    @inlinable
    static func == (lhs: Self, rhs: [Element]) -> Bool {
        for (i, element) in lhs.elements().enumerated() {
            if element != rhs[i] { return false }
        }
        return true
    }
}

public extension TensorView where Element: AnyConvertable {
    /// compares the flat elements of self with a Swift collection of elements
    @inlinable
    static func == <R>(lhs: Self, rhs: R) -> Bool
        where R: Collection, R.Element: AnyConvertable
    {
        for (lhsElement, rhsElement) in zip(lhs.elements(), rhs) {
            if lhsElement != Element(any: rhsElement) { return false }
        }
        return true
    }

    /// compares the flat elements of self with a Swift collection of elements
    @inlinable
    static func == <T>(lhs: Self, rhs: T) -> Bool where T: AnyConvertable
    {
        lhs.element == Element(any: rhs)
    }

    /// compares the flat elements of self with a Swift collection of elements
    @inlinable
    static func == <T>(lhs: T, rhs: Self) -> Bool where T: AnyConvertable
    {
        Element(any: lhs) == rhs.element
    }
}
