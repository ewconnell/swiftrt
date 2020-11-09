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
/// DiscreteStorage
public final class DiscreteStorage: StorageBuffer {
  // StorageBuffer protocol properties
  public let alignment: Int
  public let byteCount: Int
  public let id: Int
  public var isReadOnly: Bool
  public var isReference: Bool
  public var isZero: Bool

  @usableFromInline var _name: String = defaultTensorName
  @inlinable public var name: String {
    get {
      _name != defaultTensorName ? _name : "\(defaultTensorName)(\(id))"
    }
    set {
      _name = newValue
      replicas.forEach { $0?.name = newValue }
    }
  }

  //------------------------------------
  // private properties
  /// replicated device memory buffers
  public var replicas: [DeviceMemory?]

  /// the last queue used to access storage
  public var lastQueue: Platform.Device.Queue?

  /// whenever a buffer write pointer is taken, the associated DeviceMemory
  /// becomes the main copy for replication. Synchronization across threads
  /// is still required for taking multiple write pointers, however
  /// this does automatically synchronize data migrations.
  /// The value will be `nil` if no access has been taken yet
  public var main: DeviceMemory?

  /// this is incremented each time a write pointer is taken
  /// all replicated buffers will stay in sync with this version
  public var mainVersion: Int

  //------------------------------------
  // testing properties
  /// testing: `true` if the last access caused the contents of the
  /// buffer to be copied
  @inlinable public var testLastAccessCopiedDeviceMemory: Bool {
    _lastAccessCopiedMemory
  }
  public var _lastAccessCopiedMemory: Bool

  //--------------------------------------------------------------------------
  // init(type:count:order:name:
  @inlinable public init<Element>(
    storedType: Element.Type,
    count: Int,
    name: String
  ) {
    _name = name
    alignment = MemoryLayout<Element>.alignment
    byteCount = MemoryLayout<Element>.size * count
    id = Platform.objectId.next
    isReadOnly = false
    isReference = false
    isZero = false
    mainVersion = -1
    _lastAccessCopiedMemory = false

    // setup replica managment
    let numDevices = platform.devices.count
    replicas = [DeviceMemory?](repeating: nil, count: numDevices)
  }

  //--------------------------------------------------------------------------
  /// `init(storedElement:name:
  public convenience init<Element>(storedElement: Element, name: String) {
    // TODO: change this to cache single scalars
    self.init(storedType: Element.self, count: 1, name: name)
    let buffer = readWrite(
      type: Element.self, at: 0, count: 1,
      using: Platform.syncQueue)
    buffer[0] = storedElement
  }

  //--------------------------------------------------------------------------
  /// `init(storedElement:name:
  public convenience init<Element>(
    storedElement: Element,
    name: String
  ) where Element: Numeric {
    // TODO: maybe remove this special case now that the driver
    // takes single elements directly
    self.init(storedType: Element.self, count: 1, name: name)
    let buffer = readWrite(
      type: Element.self, at: 0, count: 1,
      using: Platform.syncQueue)
    buffer[0] = storedElement
    isZero = storedElement == 0
  }

  //--------------------------------------------------------------------------
  // init(type:other:using:
  @inlinable public init<Element>(
    type: Element.Type,
    copying other: DiscreteStorage,
    using queue: Platform.Device.Queue
  ) {
    id = Platform.objectId.next
    alignment = other.alignment
    byteCount = other.byteCount
    isReadOnly = other.isReadOnly
    isReference = other.isReference
    isZero = other.isZero
    _name = other._name
    _lastAccessCopiedMemory = false
    mainVersion = -1

    // setup replica managment
    replicas = [DeviceMemory?](repeating: nil, count: other.replicas.count)

    // get the memory block to copy
    let otherMemory = other.getDeviceMemory(Element.self, queue)

    // use migrate to create a new device memory instance
    _ = migrate(type, willMutate: true, using: queue)

    diagnostic(
      .copy,
      "\(other.name) --> \(self.name) \(Element.self)[\(byteCount / MemoryLayout<Element>.size)]"
      + " on \(queue.name)",
      categories: .dataCopy)

    // copy other main to self using the current queue
    queue.copyAsync(from: otherMemory, to: replicas[queue.deviceIndex]!)
  }

  //--------------------------------------------------------------------------
  // init(buffer:order:
  @inlinable public convenience init<Element>(
    referenceTo buffer: UnsafeBufferPointer<Element>,
    name: String
  ) {
    self.init(storedType: Element.self, count: buffer.count, name: name)
    isReadOnly = true
    isReference = true
    let p = UnsafeMutableBufferPointer(mutating: buffer)
    let raw = UnsafeMutableRawBufferPointer(p)
    replicas[0] = CpuDeviceMemory(0, raw, .unified, isReference: true)
    diagnostic(
      .reference, "\(name) \(Element.self)[\(buffer.count)]",
      categories: .dataAlloc)
  }

  //--------------------------------------------------------------------------
  // init(type:buffer:order:
  @inlinable public convenience init<Element>(
    referenceTo buffer: UnsafeMutableBufferPointer<Element>,
    name: String
  ) {
    self.init(storedType: Element.self, count: buffer.count, name: name)
    isReference = true
    let raw = UnsafeMutableRawBufferPointer(buffer)
    replicas[0] = CpuDeviceMemory(0, raw, .unified, isReference: true)
    diagnostic(
      .reference, "\(name) \(Element.self)[\(buffer.count)]",
      categories: .dataAlloc)
  }

  //--------------------------------------------------------------------------
  /// waitForCompletion
  /// blocks the caller until pending write operations have completed
  @inlinable public func waitForCompletion() {
    lastQueue?.waitForCompletion()
  }

  //--------------------------------------------------------------------------
  //
  @inlinable public init<S, Stream>(
    block shape: S,
    bufferedBlocks: Int,
    stream: Stream
  ) where S: TensorShape, Stream: BufferStream {
    fatalError()
  }

  //--------------------------------------------------------------------------
  // ensure that all pending work is complete before releasing memory
  @inlinable deinit {
    waitForCompletion()
  }

  //--------------------------------------------------------------------------
  //
  @inlinable public func read<Element>(
    type: Element.Type,
    at index: Int,
    count: Int,
    using queue: Platform.Device.Queue
  ) -> UnsafeBufferPointer<Element> {
    let buffer = migrate(type, willMutate: false, using: queue)
    assert(index + count <= buffer.count, "range is out of bounds")
    let start = buffer.baseAddress!.advanced(by: index)
    return UnsafeBufferPointer(start: start, count: count)
  }

  //--------------------------------------------------------------------------
  //
  @inlinable public func readWrite<Element>(
    type: Element.Type,
    at index: Int,
    count: Int,
    using queue: Platform.Device.Queue
  ) -> UnsafeMutableBufferPointer<Element> {
    let buffer = migrate(type, willMutate: true, using: queue)
    assert(index + count <= buffer.count, "range is out of bounds")
    let start = buffer.baseAddress!.advanced(by: index)
    return UnsafeMutableBufferPointer(start: start, count: count)
  }

  //--------------------------------------------------------------------------
  // getDeviceMemory
  // Manages an array of replicated device memory indexed by the deviceId
  // associated with `queue`. It will lazily create device memory if needed
  @inlinable public func getDeviceMemory<Element>(
    _ type: Element.Type,
    _ queue: Platform.Device.Queue
  ) -> DeviceMemory {
    if let memory = replicas[queue.deviceIndex] {
      return memory
    } else {
      // allocate the buffer for the target device
      // and save in the replica list
      let memory = queue.allocate(byteCount)
      replicas[queue.deviceIndex] = memory

      if willLog(level: .diagnostic) {
        let count = byteCount / MemoryLayout<Element>.size
        let msg = "\(name) on \(queue.name)  \(Element.self)[\(count)]"
        diagnostic(.alloc, msg, categories: .dataAlloc)
        memory.name = name
        memory.releaseMessage = msg
      }
      return memory
    }
  }

  //--------------------------------------------------------------------------
  /// synchronize(queue:other:willMutate)
  /// - Parameters:
  ///  - queue: the queue to synchronize
  ///  - dependentQueue: the dependent queue
  ///  - willMutate: `true` if the subsequent operation will mutate the
  ///    the tensor. Used only for diagnostics
  @inlinable public func synchronize(
    _ queue: Platform.Device.Queue,
    with dependentQueue: Platform.Device.Queue,
    _ willMutate: Bool
  ) {
    // if the queue ids are equal or the dependent queue is synchronous,
    // then the data is already implicitly synchronized
    guard queue.id != dependentQueue.id && dependentQueue.mode == .async
    else { return }

    diagnostic(
      .sync,
      "\(queue.name) will wait for" + " \(dependentQueue.name) to "
        + "\(willMutate ? "write" : "read") \(name)",
      categories: .queueSync)

    if queue.mode == .sync {
      // if the destination is synchronous, then wait for
      // `dependentQueue` to finish
      dependentQueue.waitForCompletion()
    } else {
      // if both queues are async, then record an event on the
      // the `dependentQueue` and have `queue` wait for it
      queue.wait(for: dependentQueue.recordEvent())
    }
  }

  //--------------------------------------------------------------------------
  /// migrate(type:readOnly:queue:
  /// returns a buffer on the device associated with `queue`, lazily
  /// allocating it if it does not exist. The buffer contents will match
  /// the contents of the main version (most recently mutated).
  ///
  /// - Parameters:
  ///  - type: the `Element` type
  ///  - willMutate: `true` if the returned buffer will be mutated
  ///  - queue: the queue that the returned buffer will be used
  ///
  /// - Returns: a buffer pointer to the data
  @inlinable public func migrate<Element>(
    _ type: Element.Type,
    willMutate: Bool,
    using queue: Platform.Device.Queue
  ) -> UnsafeMutableBufferPointer<Element> {
    assert(willMutate || main != nil, "attempting to read uninitialized memory")

    // For this tensor, get a buffer on the device associated
    // with `queue`. This is a synchronous operation. If the buffer
    // doesn't exist, then it will be created.
    let replica = getDeviceMemory(type, queue)

    // If there is a `main` and the replica version doesn't match
    // then we need copy main --> replica
    if let main = main, let lastQueue = lastQueue {

      func outputCopyMessage() {
        diagnostic(
          .copy,
          "\(name) dev:\(main.deviceIndex)\(setText(" --> ", color: .blue))"
            + "\(queue.name)  \(Element.self)[\(replica.count(of: Element.self))]",
          categories: .dataCopy)
      }

      func copyIfChanged(using q: Platform.Device.Queue) {
        if main.version != replica.version {
          q.copyAsync(from: main, to: replica)
          outputCopyMessage()
          // set `true` for unit test purposes
          _lastAccessCopiedMemory = true
        }
      }

      switch (main.type, replica.type) {
      // host --> host
      case (.unified, .unified):
        // no copy needed
        synchronize(queue, with: lastQueue, willMutate)

      // host --> discrete
      case (.unified, .discrete):
        synchronize(queue, with: lastQueue, willMutate)
        copyIfChanged(using: queue)

      // discrete --> host
      case (.discrete, .unified):
        copyIfChanged(using: lastQueue)
        synchronize(queue, with: lastQueue, willMutate)

      // discrete --> discrete
      case (.discrete, .discrete):
        synchronize(queue, with: lastQueue, willMutate)
        copyIfChanged(using: queue)
      }
    }

    // increment version if mutating
    if willMutate {
      mainVersion += 1
      main = replica
    }

    // the replica version either matches the main by copying
    // or is the new main
    replica.version = mainVersion

    // store a reference to the accessing queue for safe shutdown
    lastQueue = queue

    // bind to the element type
    return replica.buffer.bindMemory(to: Element.self)
  }
}
