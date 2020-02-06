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

// @Target(type:"cpu", appliedTo:"CpuAsynchronousQueue", protocol: DeviceFunctions)
public final class CpuAsynchronousQueue:
    DeviceQueue, CpuQueueProtocol, LocalDeviceQueue
{
	// protocol properties
    public var trackingId: Int
    public var defaultQueueEventOptions: QueueEventOptions
	public let device: ComputeDevice
    public let id: Int
	public let name: String
    public var logInfo: LogInfo
    public var timeout: TimeInterval?
    public var executeSynchronously: Bool = true
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    
    /// used to detect accidental queue access by other threads
    @usableFromInline
    let creatorThread: Thread
    /// the queue used for command execution
    @usableFromInline
    let commandQueue: DispatchQueue

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(logInfo: LogInfo, device: ComputeDevice, name: String, id: Int)
    {
        // create serial command queue
        commandQueue = DispatchQueue(label: "\(name).commandQueue")
        
        // create a completion event
        self.logInfo = logInfo
        self.device = device
        self.id = id
        self.name = name
        self.creatorThread = Thread.current
        defaultQueueEventOptions = QueueEventOptions()
        let path = logInfo.namePath
        trackingId = 0
        trackingId = ObjectTracker.global
            .register(self, namePath: path, isStatic: true)
        
        diagnostic("\(createString) DeviceQueue(\(trackingId)) " +
            "\(device.name)_\(name)", categories: .queueAlloc)
    }
    
    //--------------------------------------------------------------------------
    /// deinit
    /// waits for the queue to finish
    @inlinable
    deinit {
        assert(Thread.current === creatorThread,
               "Queue has been captured and is being released by a " +
            "different thread. Probably by a queued function on the queue.")

        diagnostic("\(releaseString) DeviceQueue(\(trackingId)) " +
            "\(device.name)_\(name)", categories: [.queueAlloc])
        
        // release
        ObjectTracker.global.remove(trackingId: trackingId)

        // wait for the command queue to complete before shutting down
        do {
            try waitUntilQueueIsComplete()
        } catch {
            if let timeout = self.timeout {
                diagnostic("\(timeoutString) DeviceQueue(\(trackingId)) " +
                        "\(device.name)_\(name) timeout: \(timeout)",
                        categories: [.queueAlloc])
            }
        }
    }
    
    //==========================================================================
    // generic helpers
    // generic map 1
    @inlinable
    public func mapOp<T, R>(_ x: T, _ result: inout R,
                            _ op: @escaping (T.Element) -> R.Element) where
        T: TensorView, R: TensorView
    {
        queue(#function, { x.elements(using: self) }, &result) {
            $0.map(into: &$1, op)
        }
    }
    
    // generic map 2
    @inlinable
    public func mapOp<LHS, RHS, R>(
        _ lhs: LHS, _ rhs: RHS, _ result: inout R,
        _ op: @escaping (LHS.Element, RHS.Element) -> R.Element) where
        LHS: TensorView, RHS: TensorView, R: TensorView
    {
        queue(#function, { (lhs.elements(using: self),
                            rhs.elements(using: self)) }, &result)
        {
            zip($0.0, $0.1).map(into: &$1, op)
        }
    }
    
    // generic map 3
    @inlinable
    public func mapOp<T1, T2, T3, R>(
        _ a: T1, _ b: T2, _ c: T3, _ result: inout R,
        _ op: @escaping (T1.Element, T2.Element, T3.Element) -> R.Element) where
        T1: TensorView, T2: TensorView, T3: TensorView, R: TensorView
    {
        queue(#function, { (a.elements(using: self),
                            b.elements(using: self),
                            c.elements(using: self)) }, &result)
        {
            zip($0.0, $0.1, $0.2).map(into: &$1, op)
        }
    }
    
    /// generic mapOp 3R2
    @inlinable
    public func mapOp<T1, T2, T3, R>(
        _ a: T1, _ b: T2, _ c: T3, _ result1: inout R,  _ result2: inout R,
        _ op: @escaping (T1.Element, T2.Element, T3.Element) -> (R.Element, R.Element))
        where T1: TensorView, T2: TensorView, T3: TensorView, R: TensorView
    {
        queue(#function,
              { (a.elements(using: self),
                 b.elements(using: self),
                 c.elements(using: self)) },
              { (result1.mutableElements(),
                 result2.mutableElements()) })
        { e, r in
            for ((av, bv, cv), (i0, i1)) in
                zip(zip(e.0, e.1, e.2), zip(r.0.indices, r.1.indices))
            {
                let (rv0, rv1) = op(av, bv, cv)
                r.0[i0] = rv0
                r.1[i1] = rv1
            }
        }
    }

    //--------------------------------------------------------------------------
    // does an in place op
    @inlinable
    public func inPlaceOp<T>(_ result: inout T,
                             _ op: @escaping (T.Element) -> T.Element) where
        T: MutableCollection
    {
        queue(#function, { }, { result }) { _, results in
            results.indices.forEach {
                results[$0] = op(results[$0])
            }
        }
    }

    //--------------------------------------------------------------------------
    // does a reduction op
    @inlinable
    public func reductionOp<T, R>(
        _ x: T, _ result: inout R,
        _ op: @escaping (R.Element, T.Element) -> R.Element) where
        T: Collection, R: MutableCollection
    {
        queue(#function, { x }, { result }) { elements, resultElements in
            zip(elements, resultElements.indices).forEach {
                resultElements[$1] = op(resultElements[$1], $0)
            }
        }
    }

    //--------------------------------------------------------------------------
    /// queues a closure on the queue for execution
    /// This will catch and propagate the last asynchronous error thrown.
    ///
    @inlinable
    public func queue<Inputs, R>(
        _ functionName: @autoclosure () -> String,
        _ inputs: () -> Inputs,
        _ result: inout R,
        _ body: @escaping (Inputs, inout R.MutableValues)
        -> Void) where R: TensorView
    {
        // if the queue is in an error state, no additional work is queued
        guard lastError == nil else { return }
        
        // schedule the work
        diagnostic("\(schedulingString): \(functionName())",
            categories: .scheduling)
        
        // get the parameter sequences
        let input = inputs()
        var results = result.mutableElements(using: self)
        
        if executeSynchronously {
            body(input, &results)
        } else {
            // queue the work
            commandQueue.async {
                body(input, &results)
            }
            diagnostic("\(schedulingString): \(functionName()) complete",
                categories: .scheduling)
        }
    }

    //--------------------------------------------------------------------------
    /// queues a closure on the queue for execution
    /// This will catch and propagate the last asynchronous error thrown.
    @inlinable
    public func queue<Inputs, Outputs>(
        _ functionName: @autoclosure () -> String,
        _ inputs: () throws -> Inputs,
        _ outputs: () throws -> Outputs,
        _ body: @escaping (Inputs, inout Outputs) -> Void)
    {
        // if the queue is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }
        
        // schedule the work
        diagnostic("\(schedulingString): \(functionName())",
            categories: .scheduling)
        
        // get the parameter sequences
        do {
            let input = try inputs()
            var output = try outputs()

            if executeSynchronously {
                body(input, &output)
            } else {
                // queue the work
                commandQueue.async {
                    body(input, &output)
                }
                diagnostic("\(schedulingString): \(functionName()) complete",
                    categories: .scheduling)
            }
        } catch {
            self.report(error)
        }
    }
    
    //--------------------------------------------------------------------------
    /// queues a closure on the queue for execution
    /// This will catch and propagate the last asynchronous error thrown.
    @usableFromInline
    func queue(body: @escaping () throws -> Void) {
        // if the queue is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }
        let errorDevice = device
        
        // make sure not to capture `self`
        func performBody() {
            do {
                try body()
            } catch {
                errorDevice.report(error)
            }
        }
        
        // queue the work
        if executeSynchronously {
            performBody()
        } else {
            commandQueue.async { performBody() }
        }
    }
    //--------------------------------------------------------------------------
    /// createEvent
    /// creates an event object used for queue synchronization
    @inlinable
    public func createEvent(options: QueueEventOptions) throws -> QueueEvent {
        let event = CpuAsyncEvent(options: options, timeout: timeout)
        diagnostic("\(createString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueAlloc)
        return event
    }
    
    //--------------------------------------------------------------------------
    /// record(event:
    @inlinable
    @discardableResult
    public func record(event: QueueEvent) throws -> QueueEvent {
        guard lastError == nil else { throw lastError! }
        let event = event as! CpuAsyncEvent
        diagnostic("\(recordString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
        
        // set event time
        if defaultQueueEventOptions.contains(.timing) {
            event.recordedTime = Date()
        }
        
        queue {
            event.signal()
        }
        return event
    }

    //--------------------------------------------------------------------------
    /// wait(for event:
    /// waits until the event has occurred
    @inlinable
    public func wait(for event: QueueEvent) throws {
        guard lastError == nil else { throw lastError! }
        guard !event.occurred else { return }
        diagnostic("\(waitString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
        
        queue {
            try event.wait()
        }
    }

    //--------------------------------------------------------------------------
    /// waitUntilQueueIsComplete
    /// blocks the calling thread until the command queue is empty
    @inlinable
    public func waitUntilQueueIsComplete() throws {
        let event = try record(event: createEvent())
        diagnostic("\(waitString) QueueEvent(\(event.trackingId)) " +
            "waiting for \(device.name)_\(name) to complete",
            categories: .queueSync)
        try event.wait()
        diagnostic("\(signaledString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
    }
    
    //--------------------------------------------------------------------------
    /// perform indexed copy from source view to result view
    @inlinable
    public func copy<T>(from view: T, to result: inout T) where T : TensorView {
        queue(#function, { view.elements() }, &result) {
            $0.map(into: &$1) { $0 }
        }
    }

    //--------------------------------------------------------------------------
    /// copies from one device array to another
    @inlinable
    public func copyAsync(to array: DeviceArray,
                          from otherArray: DeviceArray) throws {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        assert(array.buffer.count == otherArray.buffer.count,
               "buffer sizes don't match")
        queue {
            array.buffer.copyMemory(
                from: UnsafeRawBufferPointer(otherArray.buffer))
        }
    }

    //--------------------------------------------------------------------------
    /// copies a host buffer to a device array
    @inlinable
    public func copyAsync(to array: DeviceArray,
                          from hostBuffer: UnsafeRawBufferPointer) throws
    {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        assert(hostBuffer.baseAddress != nil)
        assert(array.buffer.count == hostBuffer.count,
               "buffer sizes don't match")
        queue {
            array.buffer.copyMemory(from: hostBuffer)
        }
    }
    
    //--------------------------------------------------------------------------
    /// copies a device array to a host buffer
    @inlinable
    public func copyAsync(to hostBuffer: UnsafeMutableRawBufferPointer,
                          from array: DeviceArray) throws
    {
        assert(hostBuffer.baseAddress != nil)
        assert(array.buffer.count == hostBuffer.count,
               "buffer sizes don't match")
        queue {
            hostBuffer.copyMemory(from: UnsafeRawBufferPointer(array.buffer))
        }
    }

    //--------------------------------------------------------------------------
    /// fills the device array with zeros
    @inlinable
    public func zero(array: DeviceArray) throws {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        queue {
            array.buffer.initializeMemory(as: UInt8.self, repeating: 0)
        }
    }
    
    //--------------------------------------------------------------------------
    /// simulateWork(x:timePerElement:result:
    /// introduces a delay in the queue by sleeping a duration of
    /// x.count * timePerElement
    public func simulateWork<T>(x: T, timePerElement: TimeInterval,
                                result: inout T)
        where T: TensorView
    {
        let delay = TimeInterval(x.count) * timePerElement
        delayQueue(atLeast: delay)
    }

    //--------------------------------------------------------------------------
    /// delayQueue(atLeast:
    /// causes the queue to sleep for the specified interval for testing
    public func delayQueue(atLeast interval: TimeInterval) {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        queue {
            Thread.sleep(forTimeInterval: interval)
        }
    }
    
    //--------------------------------------------------------------------------
    /// throwTestError
    /// used for unit testing
    public func throwTestError() {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        queue {
            throw DeviceError.queueError(idPath: [], message: "testError")
        }
    }
}

//==============================================================================
// functions that don't operate on elements
public extension CpuAsynchronousQueue {
    @inlinable
    func concat<T>(tensors: [T], alongAxis axis: Int, result: inout T) where
        T: TensorView
    {
        let inputs: () -> ([TensorValueCollection<T>]) = {
            tensors.map { $0.elements(using: self) }
        }
        
        let outputs: () throws -> ([TensorMutableValueCollection<T>]) = {
            var index = T.Shape.zeros
            var outCollections = [TensorMutableValueCollection<T>]()
            
            for tensor in tensors {
                var view = result.mutableView(at: index, extents: tensor.extents)
                outCollections.append(view.mutableElements(using: self))
                index[axis] += tensor.extents[axis]
            }
            return outCollections
        }
        
        queue(#function, inputs, outputs) { inSeqs, outSeqs in
            for i in 0..<inSeqs.count {
                for (j, k) in zip(inSeqs[i].indices, outSeqs[i].indices) {
                    outSeqs[i][k] = inSeqs[i][j]
                }
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// fill(result:with:
    @inlinable
    func fill<T>(_ result: inout T, with value: T.Element) where T: TensorView {
        queue(#function, {}, &result) { _, elements in
            elements.indices.forEach { elements[$0] = value }
        }
    }
    
    //--------------------------------------------------------------------------
    /// fillWithIndex(x:startAt:
    @inlinable
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Element: AnyNumeric
    {
        queue(#function, {}, &result) { _, elements in
            zip(elements.indices, startAt..<startAt + elements.count).forEach {
                elements[$0] = T.Element(any: $1)
            }
        }
    }
}
