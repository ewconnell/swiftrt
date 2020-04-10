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

// gyb utility docs
// https://nshipster.com/swift-gyb/

//******************************************************************************
//
// DO NOT EDIT. THIS FILE IS GENERATED FROM .swift.gyb file
//
//******************************************************************************

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
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape1.Tuple
) -> Tensor1<DType> {
   Tensor1<DType>(repeating: value, to: Shape1(shape))
}

// specifying type
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
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape2.Tuple
) -> Tensor2<DType> {
   Tensor2<DType>(repeating: value, to: Shape2(shape))
}

// specifying type
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
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape3.Tuple
) -> Tensor3<DType> {
   Tensor3<DType>(repeating: value, to: Shape3(shape))
}

// specifying type
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
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape4.Tuple
) -> Tensor4<DType> {
   Tensor4<DType>(repeating: value, to: Shape4(shape))
}

// specifying type
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
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape5.Tuple
) -> Tensor5<DType> {
   Tensor5<DType>(repeating: value, to: Shape5(shape))
}

// specifying type
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
@inlinable public func repeating(
    _ value: DType,
    _ shape: Shape6.Tuple
) -> Tensor6<DType> {
   Tensor6<DType>(repeating: value, to: Shape6(shape))
}

// specifying type
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
@inlinable public func repeating<S,E>(
    _ value: E,
    like prototype: Tensor<S,E>
) -> Tensor<S,E> where S: TensorShape
{
    Tensor<S,E>(repeating: value, to: prototype.shape)
}

// different type same shape
@inlinable public func repeating<S,E, Element>(
    _ value: Element,
    like prototype: Tensor<S,E>,
    dtype: Element.Type
) -> Tensor<S, Element> where S: TensorShape
{
    Tensor<S, Element>(repeating: value, to: prototype.shape)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
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
@inlinable public func repeating<E>(
    _ other: Tensor1<E>,
    _ shape: Shape1.Tuple
) -> Tensor1<E> {
   Tensor1<E>(repeating: other, to: Shape1(shape))
}

//---------------------------------------
// Rank2
// default type
@inlinable public func repeating<E>(
    _ other: Tensor2<E>,
    _ shape: Shape2.Tuple
) -> Tensor2<E> {
   Tensor2<E>(repeating: other, to: Shape2(shape))
}

//---------------------------------------
// Rank3
// default type
@inlinable public func repeating<E>(
    _ other: Tensor3<E>,
    _ shape: Shape3.Tuple
) -> Tensor3<E> {
   Tensor3<E>(repeating: other, to: Shape3(shape))
}

//---------------------------------------
// Rank4
// default type
@inlinable public func repeating<E>(
    _ other: Tensor4<E>,
    _ shape: Shape4.Tuple
) -> Tensor4<E> {
   Tensor4<E>(repeating: other, to: Shape4(shape))
}

//---------------------------------------
// Rank5
// default type
@inlinable public func repeating<E>(
    _ other: Tensor5<E>,
    _ shape: Shape5.Tuple
) -> Tensor5<E> {
   Tensor5<E>(repeating: other, to: Shape5(shape))
}

//---------------------------------------
// Rank6
// default type
@inlinable public func repeating<E>(
    _ other: Tensor6<E>,
    _ shape: Shape6.Tuple
) -> Tensor6<E> {
   Tensor6<E>(repeating: other, to: Shape6(shape))
}

