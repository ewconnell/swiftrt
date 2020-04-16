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
    // generatorOp
    @inlinable
    func generatorOp<R>(_ r: inout R,_ op: @escaping () -> R.Element)
        where R: MutableCollection
    {
        r.indices.forEach { r[$0] = op() }
    }

    // inPlaceOp
    @inlinable
    func inPlaceOp<R>(_ r: inout R,_ op: @escaping (R.Element) -> R.Element)
        where R: MutableCollection
    {
        r.indices.forEach { r[$0] = op(r[$0]) }
    }

    // mapOp 1
    @inlinable
    func mapOp<T, R>(_ x: T, _ r: inout R,
                     _ op: @escaping (T.Element) -> R.Element) where
        T: Collection, R: MutableCollection
    {
        zip(r.indices, x).forEach { r[$0] = op($1) }
    }

    // mapOp 2
    @inlinable
    func mapOp<LHS, RHS, R>(
        _ lhs: LHS, _ rhs: RHS, _ r: inout R,
        _ op: @escaping (LHS.Element, RHS.Element) -> R.Element) where
        LHS: Collection, RHS: Collection, R: MutableCollection
    {
        zip(r.indices, zip(lhs, rhs)).forEach { r[$0] = op($1.0, $1.1) }
    }

    // mapOp 3
    @inlinable
    func mapOp<T1, T2, T3, R>(
        _ a: T1, _ b: T2, _ c: T3, _ r: inout R,
        _ op: @escaping (T1.Element, T2.Element, T3.Element) -> R.Element) where
        T1: Collection, T2: Collection, T3: Collection, R: MutableCollection
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
        T1: Collection, T2: Collection, T3: Collection,
        R1: MutableCollection, R2: MutableCollection
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
        where T: Collection, R: MutableCollection, R.Element == T.Element
    {
        zip(r.indices, x).forEach { r[$0] = op(r[$0], $1) }
    }

    //==========================================================================
    // basic math operations cpu implementation
    //==========================================================================
    @inlinable
    func abs<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { Swift.abs($0) }
    }

    @inlinable
    func acos<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .acos($0) }
    }

    @inlinable
    func acosh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .acosh($0) }
    }

    @inlinable
    func add<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,E>)
        where S: TensorShape, E: AdditiveArithmetic
    {
        mapOp(lhs, rhs, &result, +)
    }

    @inlinable
    func and<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>,
                _ result: inout Tensor<S,Bool>)
        where S: TensorShape
    {
        mapOp(lhs, rhs, &result) { $0 && $1 }
    }

    @inlinable
    func asin<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .asin($0) }
    }

    @inlinable
    func asinh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .asinh($0) }
    }

    @inlinable
    func atan<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .atan($0) }
    }

    @inlinable
    func atan2<S,E>(_ y: Tensor<S,E>, _ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(y, x, &result) { .atan2(y: $0, x: $1) }
    }

    @inlinable
    func atanh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .atanh($0) }
    }

    // FloatingPoint -> Integer
    @inlinable
    func cast<S, E, RE>(from buffer: Tensor<S,E>, to result: inout Tensor<S,RE>)
        where S: TensorShape, E: BinaryFloatingPoint, RE: BinaryInteger
    {
        mapOp(buffer, &result) { RE($0) }
    }

    // Integer -> FloatingPoint
    @inlinable
    func cast<S, E, RE>(from buffer: Tensor<S,E>, to result: inout Tensor<S,RE>)
        where S: TensorShape, E: BinaryInteger, RE: BinaryFloatingPoint
    {
        mapOp(buffer, &result) { RE($0) }
    }

    @inlinable
    func copy<S,E>(from x: Tensor<S,E>, to result: inout Tensor<S,E>)
        where S: TensorShape
    {
        zip(result.indices, x).forEach { result[$0] = $1 }
    }

    @inlinable
    func cos<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .cos($0) }
    }

    @inlinable
    func cosh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .cosh($0) }
    }

    @inlinable
    func delay(_ interval: TimeInterval) {
        assert(Thread.current === creatorThread, _messageQueueThreadViolation)
        Thread.sleep(forTimeInterval: interval)
    }

    @inlinable
    func div<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,E>)
        where S: TensorShape, E: AlgebraicField
    {
        mapOp(lhs, rhs, &result, /)
    }

    @inlinable
    func elementsAlmostEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                                   _ tolerance: E,
                                   _ result: inout Tensor<S,Bool>)
        where S: TensorShape, E: SignedNumeric & Comparable
    {
        mapOp(lhs, rhs, &result) { Swift.abs($0 - $1) <= tolerance }
    }

    @inlinable
    func equal<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                    _ result: inout Tensor<S,Bool>)
        where S: TensorShape, E: Equatable
    {
        mapOp(lhs, rhs, &result, ==)
    }

    @inlinable
    func erf<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .erf($0) }
    }

    @inlinable
    func erfc<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .erfc($0) }
    }

    @inlinable
    func exp<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .exp($0) }
    }

    @inlinable
    func exp2<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .exp2($0) }
    }

    @inlinable
    func exp10<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .exp10($0) }
    }

    @inlinable
    func expMinusOne<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .expMinusOne($0) }
    }

    @inlinable
    func gamma<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .gamma($0) }
    }

    @inlinable
    func greater<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                      _ result: inout Tensor<S,Bool>)
        where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result, >)
    }

    @inlinable
    func greaterOrEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,Bool>)
        where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result, >=)
    }

    @inlinable
    func hypot<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, y, &result) { .hypot($0, $1) }
    }

    @inlinable
    func less<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,Bool>)
        where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result, <)
    }

    @inlinable
    func lessOrEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,Bool>)
        where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result, <=)
    }

    @inlinable
    func log<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .log($0) }
    }

    @inlinable
    func log<S,E>(onePlus x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .log(onePlus: $0) }
    }

    @inlinable
    func log2<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .log2($0) }
    }

    @inlinable
    func log10<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .log10($0) }
    }

    @inlinable
    func logGamma<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .logGamma($0) }
    }

    @inlinable
    func max<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
               _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result) { $0 >= $1 ? $0 : $1 }
    }

    @inlinable
    func min<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                   _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Comparable
    {
        mapOp(lhs, rhs, &result) { $0 <= $1 ? $0 : $1 }
    }
    
    @inlinable
    func mul<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                  _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Numeric
    {
        mapOp(lhs, rhs, &result, *)
    }
    
    @inlinable
    func neg<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: SignedNumeric
    {
        mapOp(x, &result, -)
    }

    @inlinable
    func notEqual<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                _ result: inout Tensor<S,Bool>)
        where S: TensorShape, E: Equatable
    {
        mapOp(lhs, rhs, &result, !=)
    }

    @inlinable
    func or<S>(_ lhs: Tensor<S,Bool>, _ rhs: Tensor<S,Bool>,
               _ result: inout Tensor<S,Bool>)
        where S: TensorShape
    {
        mapOp(lhs, rhs, &result) { $0 || $1 }
    }

    @inlinable
    func pow<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>,
                  _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, y, &result) { .pow($0, $1) }
    }

    @inlinable
    func pow<S, E>(_ x: Tensor<S,E>, _ n: Int, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .pow($0, n) }
    }

    @inlinable
    func replace<S,E>(_ x: Tensor<S,E>, _ y: Tensor<S,E>,
                      _ condition: Tensor<S,Bool>,
                      _ result: inout Tensor<S,E>)
    {
        mapOp(condition, y, x, &result) { $0 ? $1 : $2 }
    }

    @inlinable
    func root<S,E>(_ x: Tensor<S,E>, _ n: Int, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .root($0, n) }
    }

    @inlinable
    func sign<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { $0 < 0 ? -1 : 1 }
    }

    @inlinable
    func signGamma<S,E>(_ x: Tensor<S,E>, _ result: inout FloatingPointSign)
        where S: TensorShape, E: Real
    {
        // TODO: don't know what to do with this as set operation
        fatalError("Not implemented")
    }

    @inlinable
    func sin<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .sin($0) }
    }

    @inlinable
    func sinh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .sinh($0) }
    }

    @inlinable
    func subtract<S,E>(_ lhs: Tensor<S,E>, _ rhs: Tensor<S,E>,
                       _ result: inout Tensor<S,E>)
        where S: TensorShape, E: AdditiveArithmetic
    {
        mapOp(lhs, rhs, &result, -)
    }

    @inlinable
    func sqrt<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .sqrt($0) }
    }

    @inlinable
    func squared<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Numeric
    {
        mapOp(x, &result) { $0 * $0 }
    }

    @inlinable func reduce<S,E>(
        _ x: Tensor<S,E>,
        _ result: inout Tensor<S,E>,
        _ opId: ReductionOp,
        _ opNext: @escaping (E, E) -> E,
        _ opFinal: ReduceOpFinal<Tensor<S,E>>?
    ) {
        // repeat result to match `x`
        // this is unusual because we intentionally are writing to
        // repeated storage for result accumulation
        var repeatedResult = Tensor<S,E>(repeating: result, to: x.shape)
        
        // do the reductions
        reductionOp(x, &repeatedResult, opNext)

        if let op = opFinal {
            inPlaceOp(&result, op)
        }
    }

    @inlinable
    func tan<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .tan($0) }
    }

    @inlinable
    func tanh<S,E>(_ x: Tensor<S,E>, _ result: inout Tensor<S,E>)
        where S: TensorShape, E: Real
    {
        mapOp(x, &result) { .tanh($0) }
    }

    //==========================================================================
    // fill with value functions
    //==========================================================================
    @inlinable func fill<S,E>(_ result: inout Tensor<S,E>, with element: E)
        where S: TensorShape
    {
        generatorOp(&result) { element }
    }

    @inlinable
    func fill<S,E,B>(_ result: inout Tensor<S,E>, with range: Range<B>)
        where S: TensorShape, E: Numeric,
        B: SignedInteger, B.Stride: SignedInteger
    {
        mapOp(range.lazy.map { E(exactly: $0)! }, &result) { $0 }
    }

    @inlinable func eye<S,E>(_ result: inout Tensor<S,E>, offset: Int)
        where S: TensorShape, E: Numeric
    {
        assert(!result.isSequential)
        generatorOp(&result) { 0 }
    }

    //==========================================================================
    // fill with random functions
    // NOTE: **** These are just place holders
    // TODO: rework all of random numbers from S4TF!!
    //==========================================================================
    @inlinable func fill<S,E>(
        randomUniform result: inout Tensor<S,E>,
        _ lower: E,
        _ upper: E,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        let scale = Double(upper - lower) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)

        generatorOp(&result) {
            E(Double(generator.next()) * scale) + lower
        }
    }

    //-------------------------------------
    @inlinable func fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: E,
        _ standardDeviation: E,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        let scale = Double(standardDeviation) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)

        generatorOp(&result) {
            E(Double(generator.next()) * scale) + mean
        }
    }

    // case where the mean and stddev are not static scalars,
    // but tensor results from previous ops
    @inlinable func fill<S,E>(
        randomNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        assert(standardDeviation.count == 1 && mean.count == 1)
        let scale = Double(standardDeviation.element) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)

        generatorOp(&result) {
            E(Double(generator.next()) * scale) + mean.element
        }
    }

    //-------------------------------------
    @inlinable func fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: E,
        _ standardDeviation: E,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        let std2x = standardDeviation * 2
        let scale = Double(standardDeviation) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)

        generatorOp(&result) {
            let a = Double(generator.next()) * scale
            return E(a).clamped(to: -std2x...std2x) + mean
        }
    }

    @inlinable func fill<S,E>(
        randomTruncatedNormal result: inout Tensor<S,E>,
        _ mean: Tensor<S,E>,
        _ standardDeviation: Tensor<S,E>,
        _ seed: RandomSeed
    ) where S: TensorShape, E: BinaryFloatingPoint
    {
        assert(standardDeviation.count == 1 && mean.count == 1)
        let std2x = standardDeviation.element * 2
        let scale = Double(standardDeviation.element) / Double(UInt64.max)
        var generator = Context.createRandomNumberGenerator(using: seed)

        generatorOp(&result) {
            let a = Double(generator.next()) * scale
            return E(a).clamped(to: -std2x...std2x) + mean.element
        }
    }

    //==========================================================================
    // Deep learning operators
    //==========================================================================
////    public func createActivation<T>(
////        x: T,
////        y: inout T,
////        mode: ActivationType,
////        nan: NanPropagation,
////        reluCeiling: Double = 0) throws -> ActivationInferring<T>
////        where T: TensorView, T.Element: ScalarElement & FloatingPoint
////    {
////        fatalError("cpu not implemented")
////    }
//    #if canImport(CCuda)
//    public func convolution<T, F>(
//        activation: ActivationType,
//        strides: T.Bounds,
//        padding: Padding,
//        dilations: T.Bounds,
//        properties: ConvolutionProperties,
//        device: PlatformDevice,
//        filterBiasBackpropQueueIndex: Int) throws -> CudaConvolution<T, F>
//        where
//        T: DifferentiableTensorView, T.Element: ScalarElement,
//        F: TensorView, F.Bounds == T.Bounds, F.Element: ScalarElement
//    {
//        fatalError("cpu convolution not implemented")
//    }
//    #endif

    //==========================================================================
    // specialized derivative implementations
    //==========================================================================
    /// vjpMinMax
    @inlinable func vjpMinMax<S,E>(
        _ x: Tensor<S,E>, _ y: Tensor<S,E>, _ scale: Tensor<S,E>,
        _ op: @escaping (E, E) -> Bool,
        _ resultTrue: inout Tensor<S,E>, _ resultFalse: inout Tensor<S,E>)
        where S: TensorShape, E: Comparable & Numeric
    {
        mapOp(x, y, scale, &resultTrue, &resultFalse) {
            op($0, $1) ? ($2, E.zero) : (E.zero, $2)
        }
    }
}
