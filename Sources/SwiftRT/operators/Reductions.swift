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
import Real

//==============================================================================
// assert messages
public let _messageTensorExtentsMismatch = "tensor extents mismatch"

////==============================================================================
///// all(x:along:)
///// Returns `true` if all values are equal to `true` along the specified
///// axes. Otherwise returns `false`. The result extent along the specified
///// axes will be 1. Rank is not reduced.
///// - Parameter x: value tensor
///// - Parameter result: the scalar tensor where the result will be written
///// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
//@inlinable
//public func all<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element == Bool
//{
//    Platform.service.all(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    public func all<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//        where T: TensorView, T.Element == Bool
//    {
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents)
//        copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)
//
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .compare, { $0 && $1 }, nil)
//        return result
//    }
//}
//
///// - Parameter along: the axes to operate on
///// - Returns: a new tensor containing the result
//public extension TensorView where Element == Bool {
//    @inlinable
//    func all(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.all(self, along: axes)
//    }
//
//    @inlinable
//    func all(along axes: Int...) -> Self { all(along: Set(axes)) }
//}
//
////==============================================================================
///// any(x:along:)
///// Returns `true` if any value is equal to `true` along the specified
///// axes. Otherwise returns `false`. The result extent along the specified
///// axes will be 1. Rank is not reduced.
///// - Parameter x: value tensor
///// - Parameter along: the axes to operate on
///// - Returns: a new tensor containing the result
//@inlinable
//public func any<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element == Bool
//{
//    Platform.service.any(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    public func any<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//        where T: TensorView, T.Element == Bool
//    {
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents)
//        copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)
//
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .compare, { $0 || $1 }, nil)
//        return result
//    }
//}
//
///// - Parameter along: the axes to operate on
///// - Returns: a new tensor containing the result
//public extension TensorView where Element == Bool {
//    @inlinable
//    func any(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.any(self, along: axes)
//    }
//
//    @inlinable
//    func any(along axes: Int...) -> Self { any(along: Set(axes)) }
//}
//
//==============================================================================
/// sum(x:along:
/// Sums `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
public func sum<T>(_ x: T, along axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Numeric
{
    Platform.service.sum(x, along: axes)
}

extension PlatformService {
    @inlinable
    public func sum<T>(_ x: T, along axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element: Numeric
    {
        let extents = x.reductionExtents(along: axes)
        var result = x.createDense(with: extents).filled(with: T.Element.zero)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .add, +, nil)
        return result
    }

    @derivative(of: sum)
    @inlinable
    public func _vjpSum<T>(_ x: T, along axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T) where T: DifferentiableTensorView
    {
        let value = sum(x, along: axes)
        return (value, { [xext = x.extents] in $0.repeated(to: xext) })
    }
}
//
//public extension TensorView where Element: Numeric {
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func sum(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.sum(self, along: axes)
//    }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func sum(along axes: Int...) -> Self { sum(along: Set(axes)) }
//}
//
////==============================================================================
///// mean(x:along:
///// mean of `x` along the specified axes
/////
///// - Parameter x: value tensor
///// - Parameter along: the axes to operate on
//@inlinable
//public func mean<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element: AlgebraicField
//{
//    Platform.service.mean(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    public func mean<T>(_ x: T, along axes: Set<Int>?) -> T
//        where T: TensorView, T.Element: AlgebraicField
//    {
//        // the divisor is the product of the `axes` that are summed
//        let divisor = (axes?.reduce(T.Element.one) {
//            $0 * T.Element(exactly: x.extents[$1])!
//            }) ?? T.Element(exactly: x.count)!
//
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents).filled(with: T.Element.zero)
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .add, +, { $0 / divisor })
//        return result
//    }
//    @derivative(of: mean)
//    @inlinable
//    internal func _vjpMean<T>(_ x: T, along axes: Set<Int>? = nil)
//        -> (value: T, pullback: (T) -> T)
//        where T: DifferentiableTensorView, T.Element: AlgebraicField
//    {
//        let value = x.mean(along: axes)
//        let count = T.Element(exactly: x.count)!
//        return (value, { [xext = x.extents] in $0.repeated(to: xext) / count })
//    }
//}
//
//public extension TensorView where Element: AlgebraicField {
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func mean(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.mean(self, along: axes)
//    }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func mean(along axes: Int...) -> Self { mean(along: Set(axes)) }
//}
//
////==============================================================================
///// prod(x:along:
///// prod of `x` along the specified axes
///// - Parameter x: value tensor
///// - Parameter along: the axes to operate on
//@inlinable
//public func prod<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element: Numeric
//{
//    Platform.service.prod(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    public func prod<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//        where T: TensorView, T.Element: Numeric
//    {
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents).filled(with: T.Element.one)
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .mul, { $0 * $1 }, nil)
//        return result
//    }
//
//    @derivative(of: prod)
//    @inlinable
//    internal func _vjpProd<T>(_ x: T, along axes: Set<Int>? = nil)
//        -> (value: T, pullback: (T) -> T) where T: DifferentiableTensorView
//    {
//        let value = x.prod(along: axes)
//        return (value, { [xext = x.extents] in $0.repeated(to: xext) })
//    }
//}
//
//public extension TensorView where Element: Numeric {
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func prod(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.prod(self, along: axes)
//    }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func prod(along axes: Int...) -> Self { prod(along: Set(axes)) }
//}
//
////==============================================================================
///// prodNonZeros(x:along:
///// product of non zero values of `x` along the specified axes
///// - Parameter x: value tensor
///// - Parameter along: the axes to operate on
//@inlinable
//public func prodNonZeros<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element: Numeric
//{
//    Platform.service.prodNonZeros(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    public func prodNonZeros<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//        where T: TensorView, T.Element: Numeric
//    {
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents).filled(with: T.Element.one)
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .mulNonZeros,
//                            { $1 == 0 ? $0 : $0 * $1 }, nil)
//        return result
//    }
//
//    @derivative(of: prodNonZeros)
//    @inlinable
//    internal func _vjpProdNonZeros<T>(_ x: T, along axes: Set<Int>? = nil)
//        -> (value: T, pullback: (T) -> T)
//        where T: DifferentiableTensorView
//    {
//        // REVIEW: this is probably wrong
//        let value = x.prodNonZeros(along: axes)
//        return (value, { [xext = x.extents] in $0.repeated(to: xext) })
//    }
//}
//
//public extension TensorView where Element: Numeric {
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func prodNonZeros(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.prodNonZeros(self, along: axes)
//    }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func prodNonZeros(along axes: Int...) -> Self {
//        prodNonZeros(along: Set(axes))
//    }
//}
//
////==============================================================================
///// min(x:along:
///// returns the minimum element value of `x` along the specified axes
///// TODO: add optional indices
///// - Parameter x: value tensor
///// - Parameter along: the axes to operate on
//@inlinable
//@differentiable(where T: DifferentiableTensorView)
//public func min<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element: Comparable
//{
//    Platform.service.min(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    @differentiable(where T: DifferentiableTensorView)
//    public func min<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//        where T: TensorView, T.Element: Comparable
//    {
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents)
//        copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .min,
//                            { $0 <= $1 ? $0 : $1 }, nil)
//        return result
//    }
//
//    @derivative(of: min)
//    @inlinable
//    internal func _vjpMin<T>(_ x: T, along axes: Set<Int>? = nil)
//        -> (value: T, pullback: (T) -> T)
//        where T: DifferentiableTensorView, T.Element: Comparable
//    {
//        fatalError()
//    }
//}
//
//public extension TensorView where
//    Element: Numeric & Comparable & AnyElement
//{
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func min(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.min(self, along: axes)
//    }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func min(along axes: Int...) -> Self { min(along: Set(axes)) }
//}
//
////==============================================================================
///// max(x:along:
///// returns the maximum element value of `x` along the specified axes
///// - Parameter x: value tensor
///// - Parameter along: the axes to operate on
//@inlinable
//@differentiable(where T: DifferentiableTensorView)
//public func max<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element: Comparable
//{
//    Platform.service.max(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    @differentiable(where T: DifferentiableTensorView)
//    public func max<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//        where T: TensorView, T.Element: Comparable
//    {
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents)
//        copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .max,
//                            { $0 > $1 ? $0 : $1 }, nil)
//        return result
//    }
//
//    @derivative(of: max)
//    @inlinable
//    internal func _vjpMax<T>(_ x: T, along axes: Set<Int>? = nil)
//        -> (value: T, pullback: (T) -> T) where
//        T: DifferentiableTensorView, T.Element: Comparable
//    {
//        fatalError()
//    }
//}
//
//public extension TensorView where
//    Element: Numeric & Comparable & AnyElement
//{
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func max(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.max(self, along: axes)
//    }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func max(along axes: Int...) -> Self { max(along: Set(axes)) }
//}
//
////==============================================================================
///// absmax(x:along:
///// absolute max of `x` along the specified axes
///// - Parameter x: value tensor
///// - Parameter along: the axes to operate on
//@inlinable
//public func absmax<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element: SignedNumeric & Comparable
//{
//    Platform.service.absmax(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    public func absmax<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//        where T: TensorView, T.Element: SignedNumeric & Comparable
//    {
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents)
//        copy(from: x.view(at: T.Shape.zeros, extents: extents), to: &result)
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .amax,
//                            { Swift.max(Swift.abs($0), Swift.abs($1)) }, nil)
//        return result
//    }
//
//    @derivative(of: absmax)
//    @inlinable
//    internal func _vjpAbsmax<T>(_ x: T, along axes: Set<Int>? = nil)
//        -> (value: T, pullback: (T) -> T)
//        where T: DifferentiableTensorView, T.Element: SignedNumeric & Comparable
//    {
//        fatalError()
//    }
//}
//
//public extension TensorView where Element: SignedNumeric & Comparable
//{
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func absmax(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.absmax(self, along: axes)
//    }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func absmax(along axes: Int...) -> Self {
//        absmax(along: Set(axes))
//    }
//}
//
////==============================================================================
///// abssum(x:along:
///// Sums the absolute values of `x` along the specified axes
///// - Parameter x: value tensor
///// - Parameter along: the axes to operate on
//@inlinable
//@differentiable(where T: DifferentiableTensorView)
//public func abssum<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element: SignedNumeric & Comparable
//{
//    Platform.service.abssum(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    @differentiable(where T: DifferentiableTensorView)
//    public func abssum<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//        where T: TensorView, T.Element: SignedNumeric & Comparable
//    {
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents).filled(with: T.Element.zero)
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .asum,
//                            { $0 + Swift.abs($1) }, nil)
//        return result
//    }
//
//    @derivative(of: abssum)
//    @inlinable
//    internal func _vjpAbsSum<T>(_ x: T, along axes: Set<Int>? = nil)
//        -> (value: T, pullback: (T) -> T)
//        where T: DifferentiableTensorView, T.Element: SignedNumeric & Comparable
//    {
//        fatalError()
//    }
//}
//
//public extension TensorView where Element: SignedNumeric & Comparable {
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func abssum(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.abssum(self, along: axes)
//    }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func abssum(along axes: Int...) -> Self { abssum(along: Set(axes)) }
//}
//
////==============================================================================
///// sqrtSumSquares(x:along:
///// Square root of the sum `x` along the specified axes
///// - Parameter x: value tensor
///// - Parameter along: the axes to operate on
//@inlinable
//public func sqrtSumSquares<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//    where T: TensorView, T.Element: Real
//{
//    Platform.service.sqrtSumSquares(x, along: axes)
//}
//
//extension PlatformService {
//    @inlinable
//    public func sqrtSumSquares<T>(_ x: T, along axes: Set<Int>? = nil) -> T
//        where T: TensorView, T.Element: Real
//    {
//        let extents = x.reductionExtents(along: axes)
//        var result = x.createDense(with: extents).filled(with: T.Element.zero)
//        var resultBuffer = write(&result)
//        currentQueue.reduce(read(x), &resultBuffer, .sqrtSumSquares,
//                            { $0 + $1 * $1 }, { .sqrt($0) })
//        return result
//    }
//
//    @derivative(of: sqrtSumSquares)
//    @inlinable
//    internal func _vjpSqrtSumSquares<T>(_ x: T, along axes: Set<Int>? = nil)
//        -> (value: T, pullback: (T) -> T)
//        where T: DifferentiableTensorView, T.Element: Real
//    {
//        fatalError()
//    }
//}
//
//public extension TensorView where Element: Real {
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func sqrtSumSquares(along axes: Set<Int>? = nil) -> Self {
//        Platform.service.sqrtSumSquares(self, along: axes)
//    }
//
//    @differentiable(where Self: DifferentiableTensorView)
//    @inlinable
//    func sqrtSumSquares(along axes: Int...) -> Self {
//        sqrtSumSquares(along: Set(axes))
//    }
//}
