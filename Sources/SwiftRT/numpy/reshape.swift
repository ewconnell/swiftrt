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
//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor1<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank2
//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor2<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank3
//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor3<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank4
//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor4<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank5
//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor5<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

//==============================================================================
// Rank6
//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Int,
    order: StorageOrder = .C
) -> Tensor1<E> {
    Tensor1<E>(reshaping: a, to: Shape1(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<E> {
    Tensor2<E>(reshaping: a, to: Shape2(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<E> {
    Tensor3<E>(reshaping: a, to: Shape3(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<E> {
    Tensor4<E>(reshaping: a, to: Shape4(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<E> {
    Tensor5<E>(reshaping: a, to: Shape5(newShape), order: order)
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func reshape<E>(
    _ a: Tensor6<E>,
    _ newShape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<E> {
    Tensor6<E>(reshaping: a, to: Shape6(newShape), order: order)
}

