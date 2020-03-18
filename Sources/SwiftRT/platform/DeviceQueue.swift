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

//==============================================================================
/// DeviceQueue
/// a base class for creating custom device queues. The default implementation
/// synchronously executes all functions on the cpu. Developers can override
/// functions to delegate to accelerator devices.
open class DeviceQueue: Logging {
    // properties
    public let creatorThread: Thread
    public var defaultQueueEventOptions: QueueEventOptions
    public let deviceId: Int
    public let deviceName: String
    public let id: Int
    public let logInfo: LogInfo
    public let memoryType: MemoryType
    public let name: String
    
    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init(id: Int, parent logInfo: LogInfo,
                deviceId: Int, deviceName: String,
                memoryType: MemoryType)
    {
        self.id = id
        self.name = "q\(id)"
        self.logInfo = logInfo.flat(name)
        self.deviceId = deviceId
        self.deviceName = deviceName
        self.creatorThread = Thread.current
        self.defaultQueueEventOptions = QueueEventOptions()
        self.memoryType = memoryType
        
        diagnostic("\(createString) \(Self.self): \(deviceName)_\(name)",
            categories: .queueAlloc)
    }

    //==========================================================================
    // resource creation functions
    //==========================================================================
    
    // allocate
    @inlinable
    public func allocate(byteCount: Int, heapIndex: Int) throws -> DeviceMemory
    {
        // allocate a host memory buffer
        let buffer = UnsafeMutableRawBufferPointer.allocate(
            byteCount: byteCount, alignment: MemoryLayout<Double>.alignment)

        return DeviceMemory(buffer: buffer, memoryType: memoryType,
                            { buffer.deallocate() })
    }

    //--------------------------------------------------------------------------
    /// createEvent
    /// creates an event object used for queue synchronization
    @inlinable
    public func createEvent(options: QueueEventOptions) -> QueueEvent {
        let event = CpuQueueEvent(options: options)
        diagnostic("\(createString) QueueEvent(\(event.id)) on " +
            "\(deviceName)_\(name)", categories: .queueAlloc)
        return event
    }
    
    //--------------------------------------------------------------------------
    /// record(event:
    @inlinable
    @discardableResult
    public func record(event: QueueEvent) -> QueueEvent {
        diagnostic("\(recordString) QueueEvent(\(event.id)) on " +
            "\(deviceName)_\(name)", categories: .queueSync)
        
        // set event time
        if defaultQueueEventOptions.contains(.timing) {
            var timeStampedEvent = event
            timeStampedEvent.recordedTime = Date()
            return timeStampedEvent
        } else {
            return event
        }
    }
    
    //--------------------------------------------------------------------------
    /// wait(for event:
    /// waits until the event has occurred
    @inlinable
    public func wait(for event: QueueEvent) {
        guard !event.occurred else { return }
        diagnostic("\(waitString) QueueEvent(\(event.id)) on " +
            "\(deviceName)_\(name)", categories: .queueSync)
        do {
            try event.wait()
        } catch {
            // there is no recovery here
            writeLog("\(error)")
            fatalError()
        }
    }
    
    //--------------------------------------------------------------------------
    // waitUntilQueueIsComplete
    // the synchronous queue completes work as it is queued,
    // so it is always complete
    @inlinable
    public func waitUntilQueueIsComplete() { }
    
    //--------------------------------------------------------------------------
    // copyAsync
    @inlinable
    public func copyAsync(from memory: DeviceMemory, to other: DeviceMemory)
    {
        assert(memory.memoryType == .unified && other.memoryType == .unified)
        let buffer = UnsafeRawBufferPointer(memory.buffer)
        other.buffer.copyMemory(from: buffer)
    }
    
    //==========================================================================
    // map operation helpers for cpu implementations
    //==========================================================================
    // inPlaceOp
    @inlinable
    func inPlaceOp<R>(_ r: inout R,_ op: @escaping (R.Element) -> R.Element)
        where R: MutableShapedBuffer
    {
        r.indices.forEach { r[$0] = op(r[$0]) }
    }
    
    // mapOp 1
    @inlinable
    func mapOp<T, R>(_ x: T, _ r: inout R,
                     _ op: @escaping (T.Element) -> R.Element) where
        T: Collection, R: MutableShapedBuffer
    {
        zip(r.indices, x).forEach { r[$0] = op($1) }
    }
    
    // mapOp 1
    @inlinable
    func mapOp<T, R>(_ x: T, _ r: inout R,
                     _ op: @escaping (T.Element) -> R.Element) where
        T: ShapedBuffer, R: MutableShapedBuffer
    {
        zip(r.indices, x).forEach { r[$0] = op($1) }
    }
    
    // mapOp 2
    @inlinable
    func mapOp<LHS, RHS, R>(
        _ lhs: LHS, _ rhs: RHS, _ r: inout R,
        _ op: @escaping (LHS.Element, RHS.Element) -> R.Element) where
        LHS: ShapedBuffer, RHS: ShapedBuffer, R: MutableShapedBuffer
    {
        zip(r.indices, zip(lhs, rhs)).forEach { r[$0] = op($1.0, $1.1) }
    }
    
    // mapOp 3
    @inlinable
    func mapOp<T1, T2, T3, R>(
        _ a: T1, _ b: T2, _ c: T3, _ r: inout R,
        _ op: @escaping (T1.Element, T2.Element, T3.Element) -> R.Element) where
        T1: ShapedBuffer, T2: ShapedBuffer, T3: ShapedBuffer, R: MutableShapedBuffer
    {
        zip(r.indices, zip(a, zip(b, c))).forEach { r[$0] = op($1.0, $1.1.0, $1.1.1) }
    }
    
    // mapOp 3R2
    /// generically combines three tensors
    @inlinable
    func mapOp<T1, T2, T3, R1, R2>(
        _ a: T1, _ b: T2, _ c: T3, _ r1: inout R1,  _ r2: inout R2,
        _ op: @escaping
        (T1.Element, T2.Element, T3.Element) -> (R1.Element, R2.Element))
        where
        T1: ShapedBuffer, T2: ShapedBuffer, T3: ShapedBuffer,
        R1: MutableShapedBuffer, R2: MutableShapedBuffer
    {
        zip(zip(r1.indices, r2.indices), zip(a, zip(b, c))).forEach {
            let (r1v, r2v) = op($1.0, $1.1.0, $1.1.1)
            r1[$0.0] = r1v
            r2[$0.1] = r2v
        }
    }
    
    // reductionOp
    @inlinable
    func reductionOp<T, R>(
        _ x: T, _ r: inout R,
        _ op: @escaping (T.Element, T.Element) -> T.Element)
        where T: ShapedBuffer, R: MutableShapedBuffer, R.Element == T.Element
    {
        zip(r.indices, x).forEach { r[$0] = op(r[$0], $1) }
    }

    //==========================================================================
    // basic math operations cpu implementation
    //==========================================================================
    @inlinable
    func abs<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { Swift.abs($0) }
    }
    
    @inlinable
    func acos<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .acos($0) }
    }
    
    @inlinable
    func acosh<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .acosh($0) }
    }
    
    @inlinable
    func add<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(lhs, rhs, &result, +)
    }
    
    @inlinable
    func and<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool
    {
        mapOp(lhs, rhs, &result) { $0 && $1 }
    }
    
    @inlinable
    func asin<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .asin($0) }
    }
    
    @inlinable
    func asinh<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .asinh($0) }
    }
    
    @inlinable
    func atan<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .atan($0) }
    }
    
    @inlinable
    func atan2<T, R>(y: T, x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(y, x, &result) { .atan2(y: $0, x: $1) }
    }
    
    @inlinable
    func atanh<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .atanh($0) }
    }
    
    // FloatingPoint -> Integer
    @inlinable
    func cast<T, R>(from buffer: T, to result: inout R) where
        T: ShapedBuffer, T.Element: BinaryFloatingPoint,
        R: MutableShapedBuffer, R.Element: BinaryInteger
    {
        mapOp(buffer, &result) { R.Element($0) }
    }
    
    // Integer -> FloatingPoint
    @inlinable
    func cast<T, R>(from buffer: T, to result: inout R) where
        T: ShapedBuffer, T.Element: BinaryInteger,
        R: MutableShapedBuffer, R.Element: BinaryFloatingPoint
    {
        mapOp(buffer, &result) { R.Element($0) }
    }

    @inlinable
    func copy<T, R>(from x: T, to result: inout R) where
        T: ShapedBuffer,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        zip(result.indices, x).forEach { result[$0] = $1 }
    }
    
    @inlinable
    func cos<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .cos($0) }
    }
    
    @inlinable
    func cosh<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .cosh($0) }
    }
    
    @inlinable
    func delay(_ interval: TimeInterval) {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        Thread.sleep(forTimeInterval: interval)
    }
    
    @inlinable
    func div<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: AlgebraicField,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(lhs, rhs, &result, /)
    }
    
    @inlinable
    func elementsAlmostEqual<T, R>(_ lhs: T, _ rhs: T,
                                   _ tolerance: T.Element,
                                   _ result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric & Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        mapOp(lhs, rhs, &result) { Swift.abs($0 - $1) <= tolerance }
    }
    
    @inlinable
    func equal<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Equatable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        mapOp(lhs, rhs, &result, ==)
    }

    @inlinable
    func erf<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .erf($0) }
    }
    
    @inlinable
    func erfc<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .erfc($0) }
    }
    
    @inlinable
    func exp<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .exp($0) }
    }
    
    @inlinable
    func exp2<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .exp2($0) }
    }
    
    @inlinable
    func exp10<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .exp10($0) }
    }
    
    @inlinable
    func expMinusOne<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .expMinusOne($0) }
    }
    
    @inlinable
    func gamma<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .gamma($0) }
    }
    
    @inlinable
    func greater<T, R>(_ lhs: T, _ rhs: T, _ result: inout R)
        where T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        mapOp(lhs, rhs, &result, >)
    }
    
    @inlinable
    func greaterOrEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        mapOp(lhs, rhs, &result, >=)
    }
    
    @inlinable
    func hypot<T, R>(_ x: T, _ y: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, y, &result) { .hypot($0, $1) }
    }
    
    @inlinable
    func less<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        mapOp(lhs, rhs, &result, <)
    }
    
    @inlinable
    func lessOrEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        mapOp(lhs, rhs, &result, <=)
    }
    
    @inlinable
    func log<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .log($0) }
    }
    
    @inlinable
    func log<T, R>(onePlus x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .log(onePlus: $0) }
    }
    
    @inlinable
    func log2<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .log2($0) }
    }
    
    @inlinable
    func log10<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .log10($0) }
    }
    
    @inlinable
    func logGamma<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .logGamma($0) }
    }
    
    @inlinable
    func max<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(lhs, rhs, &result) { $0 >= $1 ? $0 : $1 }
    }
    
    @inlinable
    func min<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Comparable,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(lhs, rhs, &result) { $0 <= $1 ? $0 : $1 }
    }
    
    @inlinable
    func mul<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(lhs, rhs, &result, *)
    }
    
    @inlinable
    func neg<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: SignedNumeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result, -)
    }
    
    @inlinable
    func notEqual<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Equatable,
        R: MutableShapedBuffer, R.Element == Bool
    {
        mapOp(lhs, rhs, &result, !=)
    }
    
    @inlinable
    func or<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element == Bool,
        R: MutableShapedBuffer, R.Element == Bool
    {
        mapOp(lhs, rhs, &result) { $0 || $1 }
    }
    
    @inlinable
    func pow<T, R>(_ x: T, _ y: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, y, &result) { .pow($0, $1) }
    }
    
    @inlinable
    func pow<T, R>(_ x: T, _ n: Int, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .pow($0, n) }
    }
    
    @inlinable
    func replace<T, C, R>(_ x: T, _ y: T, _ condition: C,
                          _ result: inout R) where
        T: ShapedBuffer,
        C: ShapedBuffer, C.Element == Bool,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(condition, y, x, &result) { $0 ? $1 : $2 }
    }
    
    @inlinable
    func root<T, R>(_ x: T, _ n: Int, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .root($0, n) }
    }
    
    @inlinable
    func sign<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { $0 < 0 ? -1 : 1 }
    }
    
    @inlinable
    func signGamma<T>(_ x: T, _ result: inout FloatingPointSign) where
        T: ShapedBuffer, T.Element: Real
    {
        // TODO: don't know what to do with this as set operation
        fatalError("Not implemented")
    }
    
    @inlinable
    func sin<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .sin($0) }
    }
    
    @inlinable
    func sinh<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .sinh($0) }
    }
    
    @inlinable
    func subtract<T, R>(_ lhs: T, _ rhs: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: AdditiveArithmetic,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(lhs, rhs, &result, -)
    }
    
    @inlinable
    func sqrt<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .sqrt($0) }
    }
    
    @inlinable
    func squared<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Numeric,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { $0 * $0 }
    }
    
    @inlinable
    func reduce<T, R>(_ x: T,
                      _ result: inout R,
                      _ opId: ReductionOp,
                      _ opNext: @escaping (T.Element, T.Element) -> T.Element,
                      _ opFinal: ReduceOpFinal<R>?) where
        T: ShapedBuffer, T.Bounds == R.Bounds,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        // created a repeated shape for the initial results to match `x`
        let repeatedShape = result.shape.repeated(to: x.shape.bounds)

        // create a new elements collection to iterate using the `result`
        // buffer and the new repeated shape.
        var repeatedBuffer = MutableBufferElements(repeatedShape, result.pointer)

        // do the reductions
        reductionOp(x, &repeatedBuffer, opNext)

        if let op = opFinal {
            inPlaceOp(&result, op)
        }
    }

    @inlinable
    func tan<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .tan($0) }
    }
    
    @inlinable
    func tanh<T, R>(_ x: T, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, &result) { .tanh($0) }
    }
    
    //==========================================================================
    // fill functions
    //==========================================================================
    @inlinable
    func fill<Element, R>(_ result: inout R, with element: Element) where
        R: MutableShapedBuffer, R.Element == Element
    {
        inPlaceOp(&result) { _ in element }
    }
    
    @inlinable
    func fill<T, R>(_ result: inout R, with range: T) where
        T: StridedRangeExpression & Collection,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(range, &result) { $0 }
    }

    //-------------------------------------
    @inlinable
    func fill<R>(randomUniform result: inout R,
                 _ lowerBound: R.Element,
                 _ upperBound: R.Element,
                 _ seed: UInt64)
        where R: MutableShapedBuffer, R.Element: Numeric
    {
//        let scale = upperBound - lowerBound
//        let generator =
//        inPlaceOp(&result) { _ in
//            T.Element(generator.next()) * scale + lowerBound
//        }
    }

    //-------------------------------------
    @inlinable
    func fill<T, R>(randomNormal x: T, mean: T.Element,
                    standardDeviation: T.Element,
                    _ seed: UInt64, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }
    
    @inlinable
    func fill<T, R>(randomNormal x: T, mean: T, standardDeviation: T,
                    _ seed: UInt64, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }
    
    //-------------------------------------
    @inlinable
    func fill<T, R>(randomTruncatedNormal x: T,
                    mean: T.Element, standardDeviation: T.Element,
                    _ seed: UInt64, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }
    
    @inlinable
    func fill<T, R>(randomTruncatedNormal x: T, mean: T, standardDeviation: T,
                    _ seed: UInt64, _ result: inout R) where
        T: ShapedBuffer, T.Element: Real,
        R: MutableShapedBuffer, R.Element == T.Element
    {
        
    }

    //==========================================================================
    // Deep learning operators
    //==========================================================================
//    public func createActivation<T>(
//        x: T,
//        y: inout T,
//        mode: ActivationType,
//        nan: NanPropagation,
//        reluCeiling: Double = 0) throws -> ActivationInferring<T>
//        where T: TensorView, T.Element: ScalarElement & FloatingPoint
//    {
//        fatalError("cpu not implemented")
//    }

    public func convolution<T>(
        for x: T,
        yBounds: inout T.Bounds,
        filter: T,
        bias: T,
        activation: ActivationType,
        strides: T.Bounds,
        padding: Padding,
        dilations: T.Bounds,
        properties: ConvolutionProperties,
        device: ServiceDevice,
        filterBiasBackpropQueueIndex: Int) throws -> DeviceConvolution<T>
        where T: DifferentiableTensorView, T.Element: ScalarElement & Real
    {
        fatalError("cpu convolution not implemented")
    }

    //==========================================================================
    // specialized derivative implementations
    //==========================================================================
    /// vjpMinMax
    @inlinable
    func vjpMinMax<T, R>(
        _ x: T, _ y: T, _ scale: T,
        _ op: @escaping (T.Element, T.Element) -> Bool,
        _ resultTrue: inout R, _ resultFalse: inout R)
        where
        T : ShapedBuffer, T.Element : Comparable & Numeric,
        R : MutableShapedBuffer, R.Element == T.Element
    {
        mapOp(x, y, scale, &resultTrue, &resultFalse) {
            op($0, $1) ? ($2, T.Element.zero) : (T.Element.zero, $2)
        }
    }
}
