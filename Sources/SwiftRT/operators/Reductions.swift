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
import Numerics

//==============================================================================
// assert messages
public let _messageTensorExtentsMismatch = "tensor bounds mismatch"

//==============================================================================
/// all(x:along:)
/// Returns `true` if all values are equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.
/// - Parameter x: value tensor
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable
public func all<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element == Bool
{
    Platform.service.all(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    func all<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element == Bool
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds)
        copy(from: x.view(from: T.Bounds.zero, to: bounds), to: &result)
        
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .compare, { $0 && $1 }, nil)
        return result
    }
}

/// - Parameter along: the axes to operate on
/// - Returns: a new tensor containing the result
public extension TensorView where Element == Bool {
    @inlinable
    func all(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.all(self, alongAxes: axes)
    }
    
    @inlinable
    func all(alongAxes axes: Int...) -> Self { all(alongAxes: Set(axes)) }
}

//==============================================================================
/// any(x:along:)
/// Returns `true` if any value is equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
/// - Returns: a new tensor containing the result
@inlinable
public func any<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element == Bool
{
    Platform.service.any(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    func any<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element == Bool
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds)
        copy(from: x.view(from: T.Bounds.zero, to: bounds), to: &result)
        
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .compare, { $0 || $1 }, nil)
        return result
    }
}

/// - Parameter along: the axes to operate on
/// - Returns: a new tensor containing the result
public extension TensorView where Element == Bool {
    @inlinable
    func any(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.any(self, alongAxes: axes)
    }
    
    @inlinable
    func any(alongAxes axes: Int...) -> Self { any(alongAxes: Set(axes)) }
}

//==============================================================================
/// sum(x:along:
/// Sums `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
public func sum<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Numeric
{
    Platform.service.sum(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    func sum<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element: Numeric
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds).filled(with: T.Element.zero)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .add, +, nil)
        return result
    }
    
    @derivative(of: sum)
    @inlinable
    func _vjpSum<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T) where T: DifferentiableTensorView
    {
        let value = sum(x, alongAxes: axes)
        return (value, { [xext = x.bounds] in $0.repeated(to: xext) })
    }
}

public extension TensorView where Element: Numeric {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sum(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.sum(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sum(alongAxes axes: Int...) -> Self { sum(alongAxes: Set(axes)) }
}

//==============================================================================
/// mean(x:along:
/// mean of `x` along the specified axes
///
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
public func mean<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: AlgebraicField
{
    Platform.service.mean(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    func mean<T>(_ x: T, alongAxes axes: Set<Int>?) -> T
        where T: TensorView, T.Element: AlgebraicField
    {
        // the divisor is the product of the `axes` that are summed
        let divisor = (axes?.reduce(T.Element.one) {
            $0 * T.Element(exactly: x.bounds[$1])!
            }) ?? T.Element(exactly: x.count)!
        
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds).filled(with: T.Element.zero)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .add, +, { $0 / divisor })
        return result
    }
    
    @derivative(of: mean)
    @inlinable
    func _vjpMean<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: AlgebraicField
    {
        let value = x.mean(alongAxes: axes)
        let count = T.Element(exactly: x.count)!
        return (value, { [xext = x.bounds] in $0.repeated(to: xext) / count })
    }
}

public extension TensorView where Element: AlgebraicField {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func mean(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.mean(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func mean(alongAxes axes: Int...) -> Self { mean(alongAxes: Set(axes)) }
}

//==============================================================================
/// prod(x:along:
/// prod of `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
public func prod<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Numeric
{
    Platform.service.prod(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    func prod<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element: Numeric
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds).filled(with: T.Element.one)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .mul, { $0 * $1 }, nil)
        return result
    }
    
    @derivative(of: prod)
    @inlinable
    func _vjpProd<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T) where T: DifferentiableTensorView
    {
        let value = x.prod(alongAxes: axes)
        return (value, { [xext = x.bounds] in $0.repeated(to: xext) })
    }
}

public extension TensorView where Element: Numeric {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func prod(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.prod(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func prod(alongAxes axes: Int...) -> Self { prod(alongAxes: Set(axes)) }
}

//==============================================================================
/// prodNonZeros(x:along:
/// product of non zero values of `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
public func prodNonZeros<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Numeric
{
    Platform.service.prodNonZeros(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    func prodNonZeros<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element: Numeric
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds).filled(with: T.Element.one)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .mulNonZeros,
                            { $1 == 0 ? $0 : $0 * $1 }, nil)
        return result
    }
    
    @derivative(of: prodNonZeros)
    @inlinable
    internal func _vjpProdNonZeros<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView
    {
        // REVIEW: this is probably wrong
        let value = x.prodNonZeros(alongAxes: axes)
        return (value, { [xext = x.bounds] in $0.repeated(to: xext) })
    }
}

public extension TensorView where Element: Numeric {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func prodNonZeros(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.prodNonZeros(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func prodNonZeros(alongAxes axes: Int...) -> Self {
        prodNonZeros(alongAxes: Set(axes))
    }
}

//==============================================================================
/// min(x:along:
/// returns the minimum element value of `x` along the specified axes
/// TODO: add optional indices
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
//@differentiable(where T: DifferentiableTensorView)
public func min<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Comparable
{
    Platform.service.min(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func min<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element: Comparable
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds)
        copy(from: x.view(from: T.Bounds.zero, to: bounds), to: &result)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .min,
                            { $0 <= $1 ? $0 : $1 }, nil)
        return result
    }
    
    @inlinable
    @derivative(of: min)
    internal func _vjpMin<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Comparable
    {
        fatalError()
    }
}

public extension TensorView where Element: Comparable
{
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    func min(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.min(self, alongAxes: axes)
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    func min(alongAxes axes: Int...) -> Self { min(alongAxes: Set(axes)) }
}

//==============================================================================
/// max(x:along:
/// returns the maximum element value of `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
//@differentiable(where T: DifferentiableTensorView)
public func max<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Comparable
{
    Platform.service.max(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func max<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element: Comparable
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds)
        copy(from: x.view(from: T.Bounds.zero, to: bounds), to: &result)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .max,
                            { $0 > $1 ? $0 : $1 }, nil)
        return result
    }
    
    @derivative(of: max)
    @inlinable
    internal func _vjpMax<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T) where
        T: DifferentiableTensorView, T.Element: Comparable
    {
        fatalError()
    }
}

public extension TensorView where Element: Comparable
{
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func max(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.max(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func max(alongAxes axes: Int...) -> Self { max(alongAxes: Set(axes)) }
}

//==============================================================================
/// absmax(x:along:
/// absolute max of `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
public func absmax<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: SignedNumeric & Comparable
{
    Platform.service.absmax(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    func absmax<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element: SignedNumeric & Comparable
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds)
        copy(from: x.view(from: T.Bounds.zero, to: bounds), to: &result)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .amax,
                            { Swift.max(Swift.abs($0), Swift.abs($1)) }, nil)
        return result
    }
    
    @derivative(of: absmax)
    @inlinable
    internal func _vjpAbsmax<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: SignedNumeric & Comparable
    {
        fatalError()
    }
}

public extension TensorView where Element: SignedNumeric & Comparable
{
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func absmax(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.absmax(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func absmax(alongAxes axes: Int...) -> Self {
        absmax(alongAxes: Set(axes))
    }
}

//==============================================================================
/// abssum(x:along:
/// Sums the absolute values of `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
//@differentiable(where T: DifferentiableTensorView)
public func abssum<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: SignedNumeric & Comparable
{
    Platform.service.abssum(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    @differentiable(where T: DifferentiableTensorView)
    func abssum<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element: SignedNumeric & Comparable
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds).filled(with: T.Element.zero)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .asum,
                            { $0 + Swift.abs($1) }, nil)
        return result
    }
    
    @derivative(of: abssum)
    @inlinable
    internal func _vjpAbsSum<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: SignedNumeric & Comparable
    {
        fatalError()
    }
}

public extension TensorView where Element: SignedNumeric & Comparable {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func abssum(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.abssum(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func abssum(alongAxes axes: Int...) -> Self { abssum(alongAxes: Set(axes)) }
}

//==============================================================================
/// sqrtSumSquares(x:along:
/// Square root of the sum `x` along the specified axes
/// - Parameter x: value tensor
/// - Parameter along: the axes to operate on
@inlinable
public func sqrtSumSquares<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
    where T: TensorView, T.Element: Real
{
    Platform.service.sqrtSumSquares(x, alongAxes: axes)
}

public extension PlatformService {
    @inlinable
    func sqrtSumSquares<T>(_ x: T, alongAxes axes: Set<Int>? = nil) -> T
        where T: TensorView, T.Element: Real
    {
        let bounds = x.reductionBounds(alongAxes: axes)
        var result = x.createDense(with: bounds).filled(with: T.Element.zero)
        var resultBuffer = write(&result)
        currentQueue.reduce(read(x), &resultBuffer, .sqrtSumSquares,
                            { $0 + $1 * $1 }, { .sqrt($0) })
        return result
    }
    
    @derivative(of: sqrtSumSquares)
    @inlinable
    internal func _vjpSqrtSumSquares<T>(_ x: T, alongAxes axes: Set<Int>? = nil)
        -> (value: T, pullback: (T) -> T)
        where T: DifferentiableTensorView, T.Element: Real
    {
        fatalError()
    }
}

public extension TensorView where Element: Real {
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sqrtSumSquares(alongAxes axes: Set<Int>? = nil) -> Self {
        Platform.service.sqrtSumSquares(self, alongAxes: axes)
    }
    
    @differentiable(where Self: DifferentiableTensorView)
    @inlinable
    func sqrtSumSquares(alongAxes axes: Int...) -> Self {
        sqrtSumSquares(alongAxes: Set(axes))
    }
}
