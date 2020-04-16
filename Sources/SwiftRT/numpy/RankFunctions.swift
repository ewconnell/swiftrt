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
@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank2
@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank3
@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank4
@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank5
@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank6
@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
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
@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axis: Int) -> Tensor2<E> {
    Tensor2<E>(expanding: a, axes: Shape1(axis))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axes: Shape2.Tuple) -> Tensor3<E> {
    Tensor3<E>(expanding: a, axes: Shape2(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axes: Shape3.Tuple) -> Tensor4<E> {
    Tensor4<E>(expanding: a, axes: Shape3(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axes: Shape4.Tuple) -> Tensor5<E> {
    Tensor5<E>(expanding: a, axes: Shape4(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axes: Shape5.Tuple) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape5(axes))
}

//==============================================================================
// Rank2
@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor2<E>, axis: Int) -> Tensor3<E> {
    Tensor3<E>(expanding: a, axes: Shape1(axis))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor2<E>, axes: Shape2.Tuple) -> Tensor4<E> {
    Tensor4<E>(expanding: a, axes: Shape2(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor2<E>, axes: Shape3.Tuple) -> Tensor5<E> {
    Tensor5<E>(expanding: a, axes: Shape3(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor2<E>, axes: Shape4.Tuple) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape4(axes))
}

//==============================================================================
// Rank3
@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor3<E>, axis: Int) -> Tensor4<E> {
    Tensor4<E>(expanding: a, axes: Shape1(axis))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor3<E>, axes: Shape2.Tuple) -> Tensor5<E> {
    Tensor5<E>(expanding: a, axes: Shape2(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor3<E>, axes: Shape3.Tuple) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape3(axes))
}

//==============================================================================
// Rank4
@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor4<E>, axis: Int) -> Tensor5<E> {
    Tensor5<E>(expanding: a, axes: Shape1(axis))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor4<E>, axes: Shape2.Tuple) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape2(axes))
}

//==============================================================================
// Rank5
@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor5<E>, axis: Int) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape1(axis))
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
@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor2<E>, axis: Int) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape1(axis))
}

//==============================================================================
// Rank3
@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor3<E>, axis: Int) -> Tensor2<E> {
    Tensor2<E>(squeezing: a, axes: Shape1(axis))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor3<E>, axes: Shape2.Tuple) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape2(axes))
}

//==============================================================================
// Rank4
@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor4<E>, axis: Int) -> Tensor3<E> {
    Tensor3<E>(squeezing: a, axes: Shape1(axis))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor4<E>, axes: Shape2.Tuple) -> Tensor2<E> {
    Tensor2<E>(squeezing: a, axes: Shape2(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor4<E>, axes: Shape3.Tuple) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape3(axes))
}

//==============================================================================
// Rank5
@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor5<E>, axis: Int) -> Tensor4<E> {
    Tensor4<E>(squeezing: a, axes: Shape1(axis))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor5<E>, axes: Shape2.Tuple) -> Tensor3<E> {
    Tensor3<E>(squeezing: a, axes: Shape2(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor5<E>, axes: Shape3.Tuple) -> Tensor2<E> {
    Tensor2<E>(squeezing: a, axes: Shape3(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor5<E>, axes: Shape4.Tuple) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape4(axes))
}

//==============================================================================
// Rank6
@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axis: Int) -> Tensor5<E> {
    Tensor5<E>(squeezing: a, axes: Shape1(axis))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axes: Shape2.Tuple) -> Tensor4<E> {
    Tensor4<E>(squeezing: a, axes: Shape2(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axes: Shape3.Tuple) -> Tensor3<E> {
    Tensor3<E>(squeezing: a, axes: Shape3(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axes: Shape4.Tuple) -> Tensor2<E> {
    Tensor2<E>(squeezing: a, axes: Shape4(axes))
}

@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axes: Shape5.Tuple) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape5(axes))
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
) -> Tensor1<DType> {
   Tensor1<DType>(repeating: value, to: Shape1(shape))
}

// specifying type
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element,
    _ shape: Shape1.Tuple,
    dtype: Element.Type
) -> Tensor1<Element> {
    Tensor1<Element>(repeating: value, to: Shape1(shape))
}

//---------------------------------------
// Rank2
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape2.Tuple
) -> Tensor2<DType> {
   Tensor2<DType>(repeating: value, to: Shape2(shape))
}

// specifying type
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element,
    _ shape: Shape2.Tuple,
    dtype: Element.Type
) -> Tensor2<Element> {
    Tensor2<Element>(repeating: value, to: Shape2(shape))
}

//---------------------------------------
// Rank3
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape3.Tuple
) -> Tensor3<DType> {
   Tensor3<DType>(repeating: value, to: Shape3(shape))
}

// specifying type
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element,
    _ shape: Shape3.Tuple,
    dtype: Element.Type
) -> Tensor3<Element> {
    Tensor3<Element>(repeating: value, to: Shape3(shape))
}

//---------------------------------------
// Rank4
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape4.Tuple
) -> Tensor4<DType> {
   Tensor4<DType>(repeating: value, to: Shape4(shape))
}

// specifying type
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element,
    _ shape: Shape4.Tuple,
    dtype: Element.Type
) -> Tensor4<Element> {
    Tensor4<Element>(repeating: value, to: Shape4(shape))
}

//---------------------------------------
// Rank5
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape5.Tuple
) -> Tensor5<DType> {
   Tensor5<DType>(repeating: value, to: Shape5(shape))
}

// specifying type
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element,
    _ shape: Shape5.Tuple,
    dtype: Element.Type
) -> Tensor5<Element> {
    Tensor5<Element>(repeating: value, to: Shape5(shape))
}

//---------------------------------------
// Rank6
// default type
@differentiable
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape6.Tuple
) -> Tensor6<DType> {
   Tensor6<DType>(repeating: value, to: Shape6(shape))
}

// specifying type
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<Element>(
    _ value: Element,
    _ shape: Shape6.Tuple,
    dtype: Element.Type
) -> Tensor6<Element> {
    Tensor6<Element>(repeating: value, to: Shape6(shape))
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
// TODO: get help fixing AD error
// @differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<S,Element>(
    _ value: Element,
    like prototype: Tensor<S,Element>
) -> Tensor<S,Element> where S: TensorShape
{
    Tensor<S,Element>(repeating: value, to: prototype.shape)
}

// different type same shape
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<S,E, Element>(
    _ value: Element,
    like prototype: Tensor<S,E>,
    dtype: Element.Type
) -> Tensor<S,Element> where S: TensorShape
{
    Tensor<S,Element>(repeating: value, to: prototype.shape)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E,
    like prototype: Tensor<S,E>,
    shape: Shape1.Tuple
) -> Tensor<Shape1, E> where S: TensorShape
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return Tensor<Shape1, E>(repeating: value, to: Shape1(shape))
}

// Rank2
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E,
    like prototype: Tensor<S,E>,
    shape: Shape2.Tuple
) -> Tensor<Shape2, E> where S: TensorShape
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return Tensor<Shape2, E>(repeating: value, to: Shape2(shape))
}

// Rank3
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E,
    like prototype: Tensor<S,E>,
    shape: Shape3.Tuple
) -> Tensor<Shape3, E> where S: TensorShape
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return Tensor<Shape3, E>(repeating: value, to: Shape3(shape))
}

// Rank4
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E,
    like prototype: Tensor<S,E>,
    shape: Shape4.Tuple
) -> Tensor<Shape4, E> where S: TensorShape
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return Tensor<Shape4, E>(repeating: value, to: Shape4(shape))
}

// Rank5
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E,
    like prototype: Tensor<S,E>,
    shape: Shape5.Tuple
) -> Tensor<Shape5, E> where S: TensorShape
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return Tensor<Shape5, E>(repeating: value, to: Shape5(shape))
}

// Rank6
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<S,E>(
    _ value: E,
    like prototype: Tensor<S,E>,
    shape: Shape6.Tuple
) -> Tensor<Shape6, E> where S: TensorShape
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return Tensor<Shape6, E>(repeating: value, to: Shape6(shape))
}


//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element> where S: TensorShape
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return Tensor<Shape1, Element>(repeating: value, to: Shape1(shape))
}

// Rank2
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element> where S: TensorShape
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return Tensor<Shape2, Element>(repeating: value, to: Shape2(shape))
}

// Rank3
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element> where S: TensorShape
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return Tensor<Shape3, Element>(repeating: value, to: Shape3(shape))
}

// Rank4
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element> where S: TensorShape
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return Tensor<Shape4, Element>(repeating: value, to: Shape4(shape))
}

// Rank5
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element> where S: TensorShape
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return Tensor<Shape5, Element>(repeating: value, to: Shape5(shape))
}

// Rank6
@differentiable(where Element: DifferentiableElement)
@inlinable public func repeating<S,E,Element>(
    _ value: Element,
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element> where S: TensorShape
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
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
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor1<E>,
    _ shape: Shape1.Tuple
) -> Tensor1<E> {
   Tensor1<E>(repeating: other, to: Shape1(shape))
}

//---------------------------------------
// Rank2
// default type
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor2<E>,
    _ shape: Shape2.Tuple
) -> Tensor2<E> {
   Tensor2<E>(repeating: other, to: Shape2(shape))
}

//---------------------------------------
// Rank3
// default type
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor3<E>,
    _ shape: Shape3.Tuple
) -> Tensor3<E> {
   Tensor3<E>(repeating: other, to: Shape3(shape))
}

//---------------------------------------
// Rank4
// default type
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor4<E>,
    _ shape: Shape4.Tuple
) -> Tensor4<E> {
   Tensor4<E>(repeating: other, to: Shape4(shape))
}

//---------------------------------------
// Rank5
// default type
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor5<E>,
    _ shape: Shape5.Tuple
) -> Tensor5<E> {
   Tensor5<E>(repeating: other, to: Shape5(shape))
}

//---------------------------------------
// Rank6
// default type
@differentiable(where E: DifferentiableElement)
@inlinable public func repeating<E>(
    _ other: Tensor6<E>,
    _ shape: Shape6.Tuple
) -> Tensor6<E> {
   Tensor6<E>(repeating: other, to: Shape6(shape))
}

