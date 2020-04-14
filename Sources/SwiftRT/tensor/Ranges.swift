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
/// `SignedRangeExpression`
/// A signed range expression is a refinement of `RangeExpression`, but
/// allows `Bound` to be a negative number, which is resolved like a
/// Numpy bound by counting backwards from the upper bound of collection
/// relative ranges.
public protocol SignedRangeExpression {
    @_semantics("autodiff.nonvarying")
    func relativeTo<C>(_ collection: C) -> SignedRange
        where C : Collection, C.Index == Int
}

//==============================================================================
/// SignedRange
public struct SignedRange {
    public var lower: Int
    public var upper: Int
    @inlinable public init(_ lower: Int, _ upper: Int) {
        self.lower = lower
        self.upper = upper
    }
}

//==============================================================================
// signed range extensions
extension Range: SignedRangeExpression where Bound == Int
{
    @_semantics("autodiff.nonvarying")
    @inlinable public func relativeTo<C>(_ collection: C) -> SignedRange
        where C : Collection, C.Index == Int
    {
        let count = collection.count
        let lower = lowerBound < 0 ? lowerBound + count : lowerBound
        let upper = upperBound < 0 ? upperBound + count : upperBound
        return SignedRange(lower, upper)
    }
}

extension ClosedRange: SignedRangeExpression where Bound == Int
{
    @_semantics("autodiff.nonvarying")
    @inlinable public func relativeTo<C>(_ collection: C) -> SignedRange
        where C : Collection, C.Index == Int
    {
        let count = collection.count
        let lower = lowerBound < 0 ? lowerBound + count : lowerBound
        let upper = (upperBound < 0 ? upperBound + count : upperBound) + 1
        return SignedRange(lower, upper)
    }
}

extension PartialRangeFrom: SignedRangeExpression where Bound == Int
{
    @_semantics("autodiff.nonvarying")
    @inlinable public func relativeTo<C>(_ collection: C) -> SignedRange
        where C : Collection, C.Index == Int
    {
        let lower = lowerBound < 0 ? lowerBound + collection.count : lowerBound
        return SignedRange(lower, collection.count)
    }
}

extension PartialRangeUpTo: SignedRangeExpression where Bound == Int
{
    @_semantics("autodiff.nonvarying")
    @inlinable public func relativeTo<C>(_ collection: C) -> SignedRange
        where C : Collection, C.Index == Int
    {
        let upper = upperBound < 0 ? upperBound + collection.count : upperBound
        return SignedRange(0, upper)
    }
}

extension PartialRangeThrough: SignedRangeExpression where Bound == Int
{
    @_semantics("autodiff.nonvarying")
    @inlinable public func relativeTo<C>(_ collection: C) -> SignedRange
        where C : Collection, C.Index == Int
    {
        let count = collection.count
        let upper = (upperBound < 0 ? upperBound + count : upperBound) + 1
        return SignedRange(0, upper)
    }
}

extension Int: SignedRangeExpression
{
    @_semantics("autodiff.nonvarying")
    @inlinable public func relativeTo<C>(_ collection: C) -> SignedRange
        where C : Collection, C.Index == Int
    {
        let i = self < 0 ? self &+ collection.count : self
        return SignedRange(i, i + 1)
    }
}

//==============================================================================
// Open Range with negative bounds
infix operator ..<-: RangeFormationPrecedence

@inlinable public func ..<- (lower: Int, upper: Int) -> Range<Int>
{
    Range(uncheckedBounds: (lower, -upper))
}

//==============================================================================
// Closed Range with negative bounds
infix operator ...-: RangeFormationPrecedence

@inlinable public func ...- (lower: Int, upper: Int) -> ClosedRange<Int>
{
    ClosedRange(uncheckedBounds: (lower, -upper))
}

//==============================================================================
// PartialRangeUpTo/PartialRangeThrough negative
prefix operator ..<-
prefix operator ...-

public extension Int {
    @inlinable
    prefix static func ..<- (upper: Int) -> PartialRangeUpTo<Int> {
        ..<(-upper)
    }
    
    @inlinable
    prefix static func ...- (upper: Int) -> PartialRangeThrough<Int> {
        ...(-upper)
    }
}

//==============================================================================
/// ..+ operator
/// specifies range and relative upper bound to do windowed operations
infix operator ..+: RangeFormationPrecedence

public extension Int {
    @inlinable static func ..+ (lower: Int, extent: Int) -> Range<Int> {
        lower..<(lower + extent)
    }
}
