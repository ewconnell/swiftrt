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
/// StorageElement
/// tensor elements conform to `StorageElement` to enable reading, writing,
/// and operating on data types that are not native to the Swift language,
/// but are native and optimal on gpus.
public protocol StorageElement {
    associatedtype Stored
    associatedtype Value
    static func storedIndex(_ index: Int) -> Int
    static func storedCount(_ count: Int) -> Int
    static func value(from stored: Stored, at index: Int) -> Value
    static func store(value: Value, at index: Int, to stored: inout Stored)
}

public extension StorageElement {
    @inlinable static func storedIndex(_ index: Int) -> Int { index }
    @inlinable static func storedCount(_ count: Int) -> Int { count }
}

public extension StorageElement where Stored == Value {
    @inlinable static func value(from stored: Stored, at index: Int)
    -> Value { stored }
    
    @inlinable static func store(value: Value, at index: Int,
                                 to stored: inout Stored) { stored = value }
}

//==============================================================================
public protocol PackedStorageElement: StorageElement {
    static var indexShift: Int { get }
    static var maskingShift: Int { get }
    static var valueMask: Stored { get }
    static var valueMin: Value { get }
    static var valueMax: Value { get }
}

public extension PackedStorageElement
    where Value: BinaryInteger, Stored: BinaryInteger
{
    @inlinable static func packedShift(_ index: Int) -> Int {
        (index % (1 << indexShift)) << maskingShift
    }
    
    @inlinable static func storedIndex(_ index: Int) -> Int {
        index >> indexShift
    }
    
    @inlinable static func storedCount(_ count: Int) -> Int {
        var storedCount = count >> indexShift
        if storedCount << indexShift != count { storedCount += 1 }
        return storedCount
    }
    
    @inlinable static func value(from stored: Stored, at index: Int) -> Value {
        Value(stored >> packedShift(index) & valueMask)
    }
    
    @inlinable static func store(value: Value, at index: Int, to stored: inout Stored) {
        assert(value >= valueMin && value <= valueMax)
        let shiftCount = packedShift(index)
        if shiftCount == 0 {
            // init top bits with 0
            stored = Stored(value)
        } else {
            stored |= Stored(value) << shiftCount
        }
    }
}

//==============================================================================
// StorageElement types
public struct Int1: PackedStorageElement {
    public typealias Stored = UInt8
    public typealias Value = Int
    @inlinable public static var indexShift: Int { 3 }
    @inlinable public static var maskingShift: Int { 0 }
    @inlinable public static var valueMask: Stored { 0x1 }
    @inlinable public static var valueMin: Value { 0 }
    @inlinable public static var valueMax: Value { 1 }
}

public struct Int4: PackedStorageElement {
    public typealias Stored = UInt8
    public typealias Value = Int
    @inlinable public static var indexShift: Int { 1 }
    @inlinable public static var maskingShift: Int { 2 }
    @inlinable public static var valueMask: Stored { 0x0F }
    @inlinable public static var valueMin: Value { 0 }
    @inlinable public static var valueMax: Value { 15 }
}

extension Int8: StorageElement {
    public typealias Stored = Int8
    public typealias Value = Int8
}

extension UInt8: StorageElement {
    public typealias Stored = UInt8
    public typealias Value = UInt8
}

extension Float: StorageElement {
    public typealias Stored = Float
    public typealias Value = Float
}

extension Double: StorageElement {
    public typealias Stored = Double
    public typealias Value = Double
}
