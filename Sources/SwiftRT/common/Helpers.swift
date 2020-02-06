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

public typealias CStringPointer = UnsafePointer<CChar>

//==============================================================================
// Memory sizes
extension Int {
    @inlinable
    var KB: Int { self * 1024 }
    @inlinable
    var MB: Int { self * 1024 * 1024 }
    @inlinable
    var GB: Int { self * 1024 * 1024 * 1024 }
    @inlinable
    var TB: Int { self * 1024 * 1024 * 1024 * 1024 }
}

//==============================================================================
// String(timeInterval:
extension String {
    @inlinable
    public init(timeInterval: TimeInterval) {
        let milliseconds = Int(timeInterval
            .truncatingRemainder(dividingBy: 1.0) * 1000)
        let interval = Int(timeInterval)
        let seconds = interval % 60
        let minutes = (interval / 60) % 60
        let hours = (interval / 3600)
        self = String(format: "%0.2d:%0.2d:%0.2d.%0.3d",
                      hours, minutes, seconds, milliseconds)
    }
}

//==============================================================================
// almostEquals
@inlinable
public func almostEquals<T: AnyNumeric>(_ a: T, _ b: T,
                                        tolerance: Double = 0.00001) -> Bool {
    abs(a.asDouble - b.asDouble) < tolerance
}

//==============================================================================
// AtomicCounter
public final class AtomicCounter {
    // properties
    var counter: Int
    let mutex = Mutex()
    
    public var value: Int {
        get { mutex.sync { counter } }
        set { mutex.sync { counter = newValue } }
    }
    
    // initializers
    public init(value: Int = 0) {
        counter = value
    }
    
    // functions
    public func increment() -> Int {
        return mutex.sync {
            counter += 1
            return counter
        }
    }
}

//==============================================================================
/// Mutex
/// is this better named "critical section"
/// TODO: verify using a DispatchQueue is faster than a counting semaphore
/// TODO: rethink this and see if async(flags: .barrier makes sense using a
/// concurrent queue
public final class Mutex {
    // properties
    @usableFromInline
    let queue: DispatchQueue
    
    @inlinable
    public init() {
        queue = DispatchQueue(label: "Mutex")
    }
    
    // functions
    @inlinable
    func sync<R>(execute work: () throws -> R) rethrows -> R {
        try queue.sync(execute: work)
    }
}
