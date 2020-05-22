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

// gyb utility docs
// https://nshipster.com/swift-gyb/

//******************************************************************************
//
// DO NOT EDIT. THIS FILE IS GENERATED FROM .swift.gyb file
//
//******************************************************************************

//==============================================================================
/// reshape
/// Gives a new shape to an array without changing its data.
/// - Parameters:
///  - a: Array to be reshaped
///  - newShape:
///    The new shape should be compatible with the original shape.
///    If an integer, then the result will be a 1-D array of that length.
///    One shape dimension can be -1. In this case, the value is inferred
///    from the length of the array and remaining dimensions.
///  - order: {‘C’, ‘F’, ‘A’}, optional
///    Read the elements of a using this index order, and place the elements
///    into the reshaped array using this index order. ‘C’ means to
///    read / write the elements using C-like index order, with the
///    last axis index changing fastest, back to the first axis index
///    changing slowest. ‘F’ means to read / write the elements using
///    Fortran-like index order, with the first index changing fastest,
///    and the last index changing slowest. Note that the ‘C’ and ‘F’
///    options take no account of the memory layout of the underlying
///    array, and only refer to the order of indexing. ‘A’ means to
///    read / write the elements in Fortran-like index order if a is
///    Fortran contiguous in memory, C-like order otherwise.
/// - Returns: This will be a new view object if possible; otherwise,
///   it will be a copy. Note there is no guarantee of the memory
///   layout (C- or Fortran- contiguous) of the returned array.

//==============================================================================
// Rank1
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape1, E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape2, E> {
    Tensor<Shape2, E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape1, E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape3, E> {
    Tensor<Shape3, E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape1, E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape4, E> {
    Tensor<Shape4, E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape1, E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape5, E> {
    Tensor<Shape5, E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape1, E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape6, E> {
    Tensor<Shape6, E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank2
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape2, E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor<Shape1, E> {
    Tensor<Shape1, E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape2, E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape2, E> {
    Tensor<Shape2, E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape2, E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape3, E> {
    Tensor<Shape3, E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape2, E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape4, E> {
    Tensor<Shape4, E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape2, E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape5, E> {
    Tensor<Shape5, E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape2, E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape6, E> {
    Tensor<Shape6, E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank3
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape3, E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor<Shape1, E> {
    Tensor<Shape1, E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape3, E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape2, E> {
    Tensor<Shape2, E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape3, E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape3, E> {
    Tensor<Shape3, E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape3, E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape4, E> {
    Tensor<Shape4, E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape3, E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape5, E> {
    Tensor<Shape5, E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape3, E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape6, E> {
    Tensor<Shape6, E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank4
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape4, E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor<Shape1, E> {
    Tensor<Shape1, E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape4, E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape2, E> {
    Tensor<Shape2, E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape4, E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape3, E> {
    Tensor<Shape3, E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape4, E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape4, E> {
    Tensor<Shape4, E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape4, E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape5, E> {
    Tensor<Shape5, E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape4, E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape6, E> {
    Tensor<Shape6, E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank5
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape5, E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor<Shape1, E> {
    Tensor<Shape1, E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape5, E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape2, E> {
    Tensor<Shape2, E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape5, E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape3, E> {
    Tensor<Shape3, E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape5, E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape4, E> {
    Tensor<Shape4, E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape5, E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape5, E> {
    Tensor<Shape5, E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape5, E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape6, E> {
    Tensor<Shape6, E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank6
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape6, E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor<Shape1, E> {
    Tensor<Shape1, E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape6, E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape2, E> {
    Tensor<Shape2, E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape6, E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape3, E> {
    Tensor<Shape3, E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape6, E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape4, E> {
    Tensor<Shape4, E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape6, E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape5, E> {
    Tensor<Shape5, E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor<Shape6, E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor<Shape6, E> {
    Tensor<Shape6, E>(reshaping: a, to: Shape6(newShape), order: order)
}


//==============================================================================
/// expand
/// Expands the shape of a tensor by inserting a new axis that will
/// appear at the axis position in the expanded array shape
/// - Parameters:
///  - dims a: input array
///  - axis: the set of axes to expand in the new shape
///
//==============================================================================
// Rank1
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape1, E>, axis: Int) -> Tensor<Shape2,E> {
    Tensor<Shape2, E>(expanding: a, axes: Shape1(axis))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape1, E>, axes: Shape2.Tuple) -> Tensor<Shape3,E> {
    Tensor<Shape3,E>(expanding: a, axes: Shape2(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape1, E>, axes: Shape3.Tuple) -> Tensor<Shape4,E> {
    Tensor<Shape4,E>(expanding: a, axes: Shape3(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape1, E>, axes: Shape4.Tuple) -> Tensor<Shape5,E> {
    Tensor<Shape5,E>(expanding: a, axes: Shape4(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape1, E>, axes: Shape5.Tuple) -> Tensor<Shape6,E> {
    Tensor<Shape6,E>(expanding: a, axes: Shape5(axes))
}

//==============================================================================
// Rank2
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape2, E>, axis: Int) -> Tensor<Shape3,E> {
    Tensor<Shape3, E>(expanding: a, axes: Shape1(axis))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape2, E>, axes: Shape2.Tuple) -> Tensor<Shape4,E> {
    Tensor<Shape4,E>(expanding: a, axes: Shape2(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape2, E>, axes: Shape3.Tuple) -> Tensor<Shape5,E> {
    Tensor<Shape5,E>(expanding: a, axes: Shape3(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape2, E>, axes: Shape4.Tuple) -> Tensor<Shape6,E> {
    Tensor<Shape6,E>(expanding: a, axes: Shape4(axes))
}

//==============================================================================
// Rank3
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape3, E>, axis: Int) -> Tensor<Shape4,E> {
    Tensor<Shape4, E>(expanding: a, axes: Shape1(axis))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape3, E>, axes: Shape2.Tuple) -> Tensor<Shape5,E> {
    Tensor<Shape5,E>(expanding: a, axes: Shape2(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape3, E>, axes: Shape3.Tuple) -> Tensor<Shape6,E> {
    Tensor<Shape6,E>(expanding: a, axes: Shape3(axes))
}

//==============================================================================
// Rank4
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape4, E>, axis: Int) -> Tensor<Shape5,E> {
    Tensor<Shape5, E>(expanding: a, axes: Shape1(axis))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape4, E>, axes: Shape2.Tuple) -> Tensor<Shape6,E> {
    Tensor<Shape6,E>(expanding: a, axes: Shape2(axes))
}

//==============================================================================
// Rank5
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor<Shape5, E>, axis: Int) -> Tensor<Shape6,E> {
    Tensor<Shape6, E>(expanding: a, axes: Shape1(axis))
}


//==============================================================================
/// stack
/// Join a sequence of arrays along a new axis.
/// - Parameters:
///  - arrays: the arrays to stack. Each array must have the same shape
///  - axis: The axis in the result array along which the input
///    arrays are stacked.
///  - out: If provided, the destination to place the result.
///    The shape must be correct, matching that of what stack would have
///    returned if no out argument were specified.
///
//==============================================================================
// Rank1
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: [Tensor<Shape1, E>], axis: Int = 0)
    -> Tensor<Shape2, E>
{
    Tensor<Shape2, E>(stacking: arrays, axis: axis)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: Tensor<Shape1, E>..., axis: Int = 0)
    -> Tensor<Shape2, E>
{
    Tensor<Shape2, E>(stacking: arrays, axis: axis)
}

//@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(
    _ arrays: [Tensor<Shape1, E>],
    axis: Int = 0,
    out: inout Tensor<Shape2, E>
) -> Tensor<Shape2, E> {
    stack(arrays, axis: axis, into: &out)
    return out
}

//==============================================================================
// Rank2
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: [Tensor<Shape2, E>], axis: Int = 0)
    -> Tensor<Shape3, E>
{
    Tensor<Shape3, E>(stacking: arrays, axis: axis)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: Tensor<Shape2, E>..., axis: Int = 0)
    -> Tensor<Shape3, E>
{
    Tensor<Shape3, E>(stacking: arrays, axis: axis)
}

//@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(
    _ arrays: [Tensor<Shape2, E>],
    axis: Int = 0,
    out: inout Tensor<Shape3, E>
) -> Tensor<Shape3, E> {
    stack(arrays, axis: axis, into: &out)
    return out
}

//==============================================================================
// Rank3
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: [Tensor<Shape3, E>], axis: Int = 0)
    -> Tensor<Shape4, E>
{
    Tensor<Shape4, E>(stacking: arrays, axis: axis)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: Tensor<Shape3, E>..., axis: Int = 0)
    -> Tensor<Shape4, E>
{
    Tensor<Shape4, E>(stacking: arrays, axis: axis)
}

//@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(
    _ arrays: [Tensor<Shape3, E>],
    axis: Int = 0,
    out: inout Tensor<Shape4, E>
) -> Tensor<Shape4, E> {
    stack(arrays, axis: axis, into: &out)
    return out
}

//==============================================================================
// Rank4
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: [Tensor<Shape4, E>], axis: Int = 0)
    -> Tensor<Shape5, E>
{
    Tensor<Shape5, E>(stacking: arrays, axis: axis)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: Tensor<Shape4, E>..., axis: Int = 0)
    -> Tensor<Shape5, E>
{
    Tensor<Shape5, E>(stacking: arrays, axis: axis)
}

//@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(
    _ arrays: [Tensor<Shape4, E>],
    axis: Int = 0,
    out: inout Tensor<Shape5, E>
) -> Tensor<Shape5, E> {
    stack(arrays, axis: axis, into: &out)
    return out
}

//==============================================================================
// Rank5
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: [Tensor<Shape5, E>], axis: Int = 0)
    -> Tensor<Shape6, E>
{
    Tensor<Shape6, E>(stacking: arrays, axis: axis)
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(_ arrays: Tensor<Shape5, E>..., axis: Int = 0)
    -> Tensor<Shape6, E>
{
    Tensor<Shape6, E>(stacking: arrays, axis: axis)
}

//@differentiable(where E.Value: DifferentiableElement)
@inlinable public func stack<E>(
    _ arrays: [Tensor<Shape5, E>],
    axis: Int = 0,
    out: inout Tensor<Shape6, E>
) -> Tensor<Shape6, E> {
    stack(arrays, axis: axis, into: &out)
    return out
}


//==============================================================================
/// squeeze
/// Remove length one entries from the shape of a tensor
/// - Parameters:
///  - a: input array
///  - axis: the set of axes to squeeze in the shape
///
//==============================================================================
// Rank2
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape2,E>, axis: Int) -> Tensor<Shape1,E> {
    Tensor<Shape1,E>(squeezing: a, axes: Shape1(axis))
}

//==============================================================================
// Rank3
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape3,E>, axis: Int) -> Tensor<Shape2,E> {
    Tensor<Shape2,E>(squeezing: a, axes: Shape1(axis))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape3,E>, axes: Shape2.Tuple) -> Tensor<Shape1,E> {
    Tensor<Shape1,E>(squeezing: a, axes: Shape2(axes))
}

//==============================================================================
// Rank4
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape4,E>, axis: Int) -> Tensor<Shape3,E> {
    Tensor<Shape3,E>(squeezing: a, axes: Shape1(axis))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape4,E>, axes: Shape2.Tuple) -> Tensor<Shape2,E> {
    Tensor<Shape2,E>(squeezing: a, axes: Shape2(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape4,E>, axes: Shape3.Tuple) -> Tensor<Shape1,E> {
    Tensor<Shape1,E>(squeezing: a, axes: Shape3(axes))
}

//==============================================================================
// Rank5
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape5,E>, axis: Int) -> Tensor<Shape4,E> {
    Tensor<Shape4,E>(squeezing: a, axes: Shape1(axis))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape5,E>, axes: Shape2.Tuple) -> Tensor<Shape3,E> {
    Tensor<Shape3,E>(squeezing: a, axes: Shape2(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape5,E>, axes: Shape3.Tuple) -> Tensor<Shape2,E> {
    Tensor<Shape2,E>(squeezing: a, axes: Shape3(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape5,E>, axes: Shape4.Tuple) -> Tensor<Shape1,E> {
    Tensor<Shape1,E>(squeezing: a, axes: Shape4(axes))
}

//==============================================================================
// Rank6
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape6,E>, axis: Int) -> Tensor<Shape5,E> {
    Tensor<Shape5,E>(squeezing: a, axes: Shape1(axis))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape6,E>, axes: Shape2.Tuple) -> Tensor<Shape4,E> {
    Tensor<Shape4,E>(squeezing: a, axes: Shape2(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape6,E>, axes: Shape3.Tuple) -> Tensor<Shape3,E> {
    Tensor<Shape3,E>(squeezing: a, axes: Shape3(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape6,E>, axes: Shape4.Tuple) -> Tensor<Shape2,E> {
    Tensor<Shape2,E>(squeezing: a, axes: Shape4(axes))
}

@differentiable(where E.Value: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor<Shape6,E>, axes: Shape5.Tuple) -> Tensor<Shape1,E> {
    Tensor<Shape1,E>(squeezing: a, axes: Shape5(axes))
}


//==============================================================================
/// repeating
/// Return a new tensor of given shape and type repeating `value`
/// - Parameters:
///  - value: to repeat
///  - shape: Int or tuple of Int. Shape of the array, e.g., (2, 3) or 2.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
/// - Returns: read only repeated element

//---------------------------------------
// Rank1
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape1.Tuple
) -> Tensor<Shape1, DType> {
   Tensor<Shape1,DType>(repeating: value, to: Shape1(shape))
}

// specifying type
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element.Value,
    _ shape: Shape1.Tuple,
    dtype: Element.Type
) -> Tensor<Shape1,Element> {
    Tensor<Shape1, Element>(repeating: value, to: Shape1(shape))
}

//---------------------------------------
// Rank2
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape2.Tuple
) -> Tensor<Shape2, DType> {
   Tensor<Shape2,DType>(repeating: value, to: Shape2(shape))
}

// specifying type
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element.Value,
    _ shape: Shape2.Tuple,
    dtype: Element.Type
) -> Tensor<Shape2,Element> {
    Tensor<Shape2, Element>(repeating: value, to: Shape2(shape))
}

//---------------------------------------
// Rank3
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape3.Tuple
) -> Tensor<Shape3, DType> {
   Tensor<Shape3,DType>(repeating: value, to: Shape3(shape))
}

// specifying type
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element.Value,
    _ shape: Shape3.Tuple,
    dtype: Element.Type
) -> Tensor<Shape3,Element> {
    Tensor<Shape3, Element>(repeating: value, to: Shape3(shape))
}

//---------------------------------------
// Rank4
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape4.Tuple
) -> Tensor<Shape4, DType> {
   Tensor<Shape4,DType>(repeating: value, to: Shape4(shape))
}

// specifying type
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element.Value,
    _ shape: Shape4.Tuple,
    dtype: Element.Type
) -> Tensor<Shape4,Element> {
    Tensor<Shape4, Element>(repeating: value, to: Shape4(shape))
}

//---------------------------------------
// Rank5
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape5.Tuple
) -> Tensor<Shape5, DType> {
   Tensor<Shape5,DType>(repeating: value, to: Shape5(shape))
}

// specifying type
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element.Value,
    _ shape: Shape5.Tuple,
    dtype: Element.Type
) -> Tensor<Shape5,Element> {
    Tensor<Shape5, Element>(repeating: value, to: Shape5(shape))
}

//---------------------------------------
// Rank6
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape6.Tuple
) -> Tensor<Shape6, DType> {
   Tensor<Shape6,DType>(repeating: value, to: Shape6(shape))
}

// specifying type
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element.Value,
    _ shape: Shape6.Tuple,
    dtype: Element.Type
) -> Tensor<Shape6,Element> {
    Tensor<Shape6, Element>(repeating: value, to: Shape6(shape))
}


//==============================================================================
/// repeating(value:like:
/// Return a new tensor of given shape and type repeating `value`
/// - Parameters:
///  - value: to repeat
///  - prototype: attributes are copied from this tensor when not specified
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - shape: Int or tuple of Int. Shape of the array, e.g., (2, 3) or 2.
/// - Returns: read only repeated element

// same type and shape
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<S,Element>(
    _ value: Element.Value,
    like prototype: Tensor<S,Element>
) -> Tensor<S,Element> where S: TensorShape
{
    Tensor<S,Element>(repeating: value, to: prototype.shape)
}

// different type same shape
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<S,E, Element>(
    _ value: Element.Value,
    like prototype: Tensor<S,E>,
    dtype: Element.Type
) -> Tensor<S,Element> where S: TensorShape
{
    Tensor<S,Element>(repeating: value, to: prototype.shape)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E.Value,
    like prototype: Tensor<S,E>,
    shape: Shape1.Tuple
) -> Tensor<Shape1, E> where S: TensorShape
{
    assert(prototype.count == Shape1(shape).elementCount())
    return Tensor<Shape1, E>(repeating: value, to: Shape1(shape))
}

// Rank2
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E.Value,
    like prototype: Tensor<S,E>,
    shape: Shape2.Tuple
) -> Tensor<Shape2, E> where S: TensorShape
{
    assert(prototype.count == Shape2(shape).elementCount())
    return Tensor<Shape2, E>(repeating: value, to: Shape2(shape))
}

// Rank3
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E.Value,
    like prototype: Tensor<S,E>,
    shape: Shape3.Tuple
) -> Tensor<Shape3, E> where S: TensorShape
{
    assert(prototype.count == Shape3(shape).elementCount())
    return Tensor<Shape3, E>(repeating: value, to: Shape3(shape))
}

// Rank4
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E.Value,
    like prototype: Tensor<S,E>,
    shape: Shape4.Tuple
) -> Tensor<Shape4, E> where S: TensorShape
{
    assert(prototype.count == Shape4(shape).elementCount())
    return Tensor<Shape4, E>(repeating: value, to: Shape4(shape))
}

// Rank5
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E.Value,
    like prototype: Tensor<S,E>,
    shape: Shape5.Tuple
) -> Tensor<Shape5, E> where S: TensorShape
{
    assert(prototype.count == Shape5(shape).elementCount())
    return Tensor<Shape5, E>(repeating: value, to: Shape5(shape))
}

// Rank6
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E.Value,
    like prototype: Tensor<S,E>,
    shape: Shape6.Tuple
) -> Tensor<Shape6, E> where S: TensorShape
{
    assert(prototype.count == Shape6(shape).elementCount())
    return Tensor<Shape6, E>(repeating: value, to: Shape6(shape))
}


//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element.Value,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element> where S: TensorShape
{
    assert(prototype.count == Shape1(shape).elementCount())
    return Tensor<Shape1, Element>(repeating: value, to: Shape1(shape))
}

// Rank2
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element.Value,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element> where S: TensorShape
{
    assert(prototype.count == Shape2(shape).elementCount())
    return Tensor<Shape2, Element>(repeating: value, to: Shape2(shape))
}

// Rank3
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element.Value,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element> where S: TensorShape
{
    assert(prototype.count == Shape3(shape).elementCount())
    return Tensor<Shape3, Element>(repeating: value, to: Shape3(shape))
}

// Rank4
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element.Value,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element> where S: TensorShape
{
    assert(prototype.count == Shape4(shape).elementCount())
    return Tensor<Shape4, Element>(repeating: value, to: Shape4(shape))
}

// Rank5
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element.Value,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element> where S: TensorShape
{
    assert(prototype.count == Shape5(shape).elementCount())
    return Tensor<Shape5, Element>(repeating: value, to: Shape5(shape))
}

// Rank6
@differentiable(where Element.Value: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element.Value,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element> where S: TensorShape
{
    assert(prototype.count == Shape6(shape).elementCount())
    return Tensor<Shape6, Element>(repeating: value, to: Shape6(shape))
}


//==============================================================================
// repeating(other:shape:
/// - Parameters:
///  - other: the tensor to repeat
///  - shape: Int or tuple of Int. Shape of the array, e.g., (2, 3) or 2.
/// - Returns: read only tensor with `other` spatially repeated
//---------------------------------------
// Rank1
// default type
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor<Shape1,E>,
    _ shape: Shape1.Tuple
) -> Tensor<Shape1,E> {
   Tensor<Shape1,E>(repeating: other, to: Shape1(shape))
}

//---------------------------------------
// Rank2
// default type
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor<Shape2,E>,
    _ shape: Shape2.Tuple
) -> Tensor<Shape2,E> {
   Tensor<Shape2,E>(repeating: other, to: Shape2(shape))
}

//---------------------------------------
// Rank3
// default type
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor<Shape3,E>,
    _ shape: Shape3.Tuple
) -> Tensor<Shape3,E> {
   Tensor<Shape3,E>(repeating: other, to: Shape3(shape))
}

//---------------------------------------
// Rank4
// default type
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor<Shape4,E>,
    _ shape: Shape4.Tuple
) -> Tensor<Shape4,E> {
   Tensor<Shape4,E>(repeating: other, to: Shape4(shape))
}

//---------------------------------------
// Rank5
// default type
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor<Shape5,E>,
    _ shape: Shape5.Tuple
) -> Tensor<Shape5,E> {
   Tensor<Shape5,E>(repeating: other, to: Shape5(shape))
}

//---------------------------------------
// Rank6
// default type
@differentiable(where E.Value: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor<Shape6,E>,
    _ shape: Shape6.Tuple
) -> Tensor<Shape6,E> {
   Tensor<Shape6,E>(repeating: other, to: Shape6(shape))
}

