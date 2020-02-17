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
// StaticArrayProtocol
public protocol StaticArrayProtocol:
    RandomAccessCollection,
    MutableCollection,
    CustomStringConvertible
{
    /// the element type in the array
    associatedtype Element
    /// a tuple that is used for storage whose size must
    /// be a multiple of `Element` size
    associatedtype Storage
    
    /// the array storage
    var storage: Storage { get set }

    // initialzers
    init(_ data: Storage)
    init?(_ data: Storage?)
}

//==============================================================================
/// StaticArrayProtocol extension
extension StaticArrayProtocol {
    /// the collection as a Swift Array
    @inlinable
    public var array: [Element] { [Element](self) }
    /// alias used to make clear what is being handled
    @inlinable
    public var tuple: Storage { storage }
    /// the number of elements in the array
    @inlinable
    public var count: Int {
        MemoryLayout<Storage>.size / MemoryLayout<Element>.size
    }
    /// the last index value
    @inlinable
    public var lastIndex: Int { count - 1 }
    /// starting index
    @inlinable
    public var startIndex: Int { 0 }
    /// ending index
    @inlinable
    public var endIndex: Int { count }
    /// description for diagnostics
    @inlinable
    public var description: String { String(describing: array) }
}

//==============================================================================
/// StaticArray
public struct StaticArray<Element, Storage> : StaticArrayProtocol {
    public var storage: Storage
    
    //------------------------------------
    // initializers
    @inlinable
    public init(_ data: Storage) {
        assert(MemoryLayout<Storage>.size % MemoryLayout<Element>.size == 0,
               "Storage size must be multiple of Element size")
        storage = data
    }

    @inlinable
    public init?(_ data: Storage?) {
        guard let data = data else { return nil }
        self.init(data)
    }
    
    // subscript
    @inlinable
    public subscript(index: Int) -> Element {
        get {
            withUnsafeBytes(of: storage) {
                $0.bindMemory(to: Element.self)[index]
            }
        }
        set {
            withUnsafeMutableBytes(of: &storage) {
                $0.bindMemory(to: Element.self)[index] = newValue
            }
        }
    }
}

//------------------------------------------------------------------------------
// Equatable
extension StaticArray: Equatable where Element: Equatable {
    @inlinable
    public static func == (lhs: Self, rhs: Self) -> Bool {
        withUnsafeBytes(of: lhs.storage) { lhsPtr in
            withUnsafeBytes(of: rhs.storage) { rhsPtr in
                memcmp(lhsPtr.baseAddress!,
                       rhsPtr.baseAddress!,
                       MemoryLayout<Storage>.size) == 0
            }
        }
    }

    @inlinable
    public static func == (lhs: Self, rhs: [Element]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        for i in 0..<lhs.count {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }

    @inlinable
    public static func == (lhs: [Element], rhs: Self) -> Bool {
        guard lhs.count == rhs.count else { return false }
        for i in 0..<lhs.count {
            if lhs[i] != rhs[i] { return false }
        }
        return true
    }
}

//------------------------------------------------------------------------------
// Codable
extension StaticArray: Codable where Element: Codable {
    enum CodingKeys: String, CodingKey { case data }
    
    /// encodes the contents of the array
    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try forEach {
            try container.encode($0)
        }
    }
    
    // TODO: do a perf test to see if the ManagedBuffer class is faster
    // than using ContiguousArray
    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        let rank = MemoryLayout<Storage>.size / MemoryLayout<Element>.size
        var array = ContiguousArray<Int>(repeating: 0, count: rank)
        for i in 0..<rank {
            array[i] = try container.decode(Int.self)
        }
        self.init(array.withUnsafeBytes {
            $0.bindMemory(to: Storage.self)[0]
        })
    }
}

//==============================================================================
//
public extension StaticArrayProtocol {
    @inlinable
    func map(_ transform: (Element) -> Element) -> Self {
        var result = self
        zip(result.indices, self).forEach { result[$0] = transform($1) }
        return result
    }
    
    @inlinable
    func reduce<Result>(
        _ initialResult: Result,
        _ nextPartialResult: (Result, Element) -> Result) -> Result
    {
        var result = initialResult
        forEach { result = nextPartialResult(result, $0) }
        return result
    }
}
