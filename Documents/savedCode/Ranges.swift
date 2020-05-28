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
/// RangeBound
public protocol RangeBound: Comparable, Numeric {
    func steps(dividedBy step: Self) -> Int
}

public extension RangeBound where Self: FixedWidthInteger {
    @inlinable
    func steps(dividedBy step: Self) -> Int { Int(self / step) }
}

public extension RangeBound where Self: BinaryFloatingPoint {
    @inlinable
    func steps(dividedBy step: Self) -> Int { Int(self / step) }
}

extension Int: RangeBound { }
extension Float: RangeBound { }
extension Double: RangeBound { }

//==============================================================================
/// StridedRangeExpression
public protocol PartialRangeExpression {
    associatedtype Bound: RangeBound

    var step: Bound { get }

    @_semantics("autodiff.nonvarying")
    func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
}

extension PartialRangeExpression {
    @inlinable
    public var step: Bound { 1 }
}

//==============================================================================
/// PartialStridedRange
public struct PartialStridedRange<Partial>: PartialRangeExpression
    where Partial: RangeExpression, Partial.Bound: RangeBound
{
    public typealias Bound = Partial.Bound
    public var partialRange: Partial
    public var step: Bound
    
    @inlinable
    public init(partial range: Partial, by step: Bound) {
        self.partialRange = range
        self.step = step
    }
    
    @inlinable
    @_semantics("autodiff.nonvarying")
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let r = partialRange.relative(to: collection)
        return StridedRange(from: r.lowerBound, to: r.upperBound, by: step)
    }
}

//==============================================================================
/// range operators
@inlinable
public func .. (range: UnboundedRange, step: Int)
    -> PartialStridedRange<PartialRangeFrom<Int>>
{
    PartialStridedRange(partial: 0..., by: step)
}


// whole range stepped
prefix operator .....

public extension Int {
    @inlinable
    prefix static func ..... (step: Int) ->
        PartialStridedRange<PartialRangeFrom<Int>>
    {
        PartialStridedRange(partial: 0..., by: step)
    }
}

//==============================================================================
/// RelativeRange
public struct RelativeRange: RangeExpression, PartialRangeExpression {
    public typealias Bound = Int
    public var start: Int
    public var extent: Int
    
    @inlinable
    public init(from start: Int, extent: Int) {
        assert(extent > 0, "cannot specify and empty range window")
        self.start = start
        self.extent = extent
    }
    
    @inlinable
    public func relative<C>(to collection: C) -> Range<Int>
        where C : Collection, C.Index == Int
    {
        let i = start < 0 ? start &+ collection.count : start
        return Range(uncheckedBounds: (i, i &+ extent))
    }
    
    @inlinable
    public func contains(_ element: Int) -> Bool { true }
    
    @inlinable
    @_semantics("autodiff.nonvarying")
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let i = start < 0 ? start &+ collection.count : start
        return StridedRange(from: i, to: i &+ extent, by: step)
    }
    
    @inlinable
    public static func .. (range: Self, step: Bound) ->
        PartialStridedRange<Self>
    {
        PartialStridedRange(partial: range, by: step)
    }
}

//==============================================================================
/// StridedRangeExpression
public protocol StridedRangeExpression: PartialRangeExpression {
    var stridedRange: StridedRange<Bound> { get }
}

//==============================================================================
/// StridedRange
public struct StridedRange<Bound>: StridedRangeExpression, Collection
    where Bound: RangeBound
{
    // properties
    public let count: Int
    public let start: Bound
    public let end: Bound
    public let step: Bound
    public var stridedRange: StridedRange<Bound> { self }
    
    // open range init
    @inlinable
    public init(from lower: Bound, to upper: Bound, by step: Bound) {
        assert(lower < upper, "Empty range: `to` must be greater than `from`")
        self.count = (upper - lower).steps(dividedBy: step)
        self.start = lower
        self.end = upper
        self.step = step
    }
    
    // closed range init
    @inlinable
    public init(from lower: Bound, through upper: Bound, by step: Bound) {
        assert(lower <= upper,
               "Empty range: `to` must be greater than or equal to `from`")
        let rangeCount = (upper - lower + step)
        self.count = rangeCount.steps(dividedBy: step)
        self.start = lower
        self.end = lower + rangeCount
        self.step = step
    }
    
    @inlinable
    @_semantics("autodiff.nonvarying")
    public func relativeTo<C>(_ collection: C) -> Self
        where C : Collection, Bound == C.Index { self }
    
    // Collection
    @inlinable
    public var startIndex: Int { 0 }

    @inlinable
    public var endIndex: Int { count }

    @inlinable
    public subscript(position: Int) -> Bound {
        Bound(exactly: position)! * step
    }

    @inlinable
    public func index(after i: Int) -> Int { i &+ 1 }
}

//==============================================================================
/// StridedRangeExpression
extension Range: StridedRangeExpression, PartialRangeExpression
    where Bound: RangeBound
{
    @inlinable
    @_semantics("autodiff.nonvarying")
    public var stridedRange: StridedRange<Bound> {
        StridedRange(from: lowerBound, to: upperBound, by: step)
    }

    @inlinable
    @_semantics("autodiff.nonvarying")
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let start = lowerBound < 0 ? lowerBound + count : lowerBound
        let end = upperBound < 0 ? upperBound + count : upperBound
        return StridedRange(from: start, to: end, by: step)
    }
    
    @inlinable
    public static func .. (r: Self, step: Bound) -> StridedRange<Bound> {
        StridedRange(from: r.lowerBound, to: r.upperBound, by: step)
    }
}

extension ClosedRange: StridedRangeExpression, PartialRangeExpression
    where Bound: RangeBound
{
    @inlinable
    @_semantics("autodiff.nonvarying")
    public var stridedRange: StridedRange<Bound> {
        StridedRange(from: lowerBound, through: upperBound, by: step)
    }
    
    @inlinable
    @_semantics("autodiff.nonvarying")
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let start = lowerBound < 0 ? lowerBound + count : lowerBound
        let end = upperBound < 0 ? upperBound + count : upperBound
        return StridedRange(from: start, through: end, by: step)
    }

    @inlinable
    public static func .. (r: Self, step: Bound) -> StridedRange<Bound> {
        StridedRange(from: r.lowerBound, through: r.upperBound, by: step)
    }
}

extension PartialRangeFrom: PartialRangeExpression where Bound: RangeBound {
    @inlinable
    @_semantics("autodiff.nonvarying")
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let start = lowerBound < 0 ? lowerBound + count : lowerBound
        return StridedRange(from: start, to: count, by: step)
    }

    @inlinable
    public static func .. (range: Self, step: Bound) ->
        PartialStridedRange<Self>
    {
        PartialStridedRange(partial: range, by: step)
    }
}

extension PartialRangeUpTo: PartialRangeExpression where Bound: RangeBound {
    @inlinable
    @_semantics("autodiff.nonvarying")
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let end = upperBound < 0 ? upperBound + count : upperBound
        return StridedRange(from: 0, to: end, by: step)
    }

    @inlinable
    public static func .. (range: Self, step: Bound) ->
        PartialStridedRange<Self>
    {
        PartialStridedRange(partial: range, by: step)
    }
}

extension PartialRangeThrough: PartialRangeExpression
    where Bound: RangeBound
{
    @inlinable
    @_semantics("autodiff.nonvarying")
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let count = Bound(exactly: collection.count)!
        let end = (upperBound < 0 ? upperBound + count : upperBound) + step
        return StridedRange(from: 0, to: end, by: step)
    }

    @inlinable
    public static func .. (range: Self, step: Bound) ->
        PartialStridedRange<Self>
    {
        PartialStridedRange(partial: range, by: step)
    }
}

extension Int: PartialRangeExpression {
    public typealias Bound = Int
    
    @inlinable
    @_semantics("autodiff.nonvarying")
    public func relativeTo<C>(_ collection: C) -> StridedRange<Bound>
        where C : Collection, Self.Bound == C.Index
    {
        let i = self < 0 ? self &+ collection.count : self
        return StridedRange(from: i, to: i &+ 1, by: 1)
    }
}

