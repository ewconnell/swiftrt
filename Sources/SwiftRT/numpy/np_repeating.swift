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
///  - value: Fill value
///  - shape: Int or tuple of Int. Shape of the array, e.g., (2, 3) or 2.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
/// - Returns: Fill of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.
@inlinable
public func repeating<Shape, Element>(
    _ value: Element,
    _ shape: Shape.Tuple,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element> where Shape: TensorShape
{
    repeating(value, Shape(shape), dtype, order)
}

@inlinable
public func repeating<Shape, Element>(
    _ value: Element,
    _ shape: Shape,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element> where Shape: TensorShape
{
    Tensor<Shape, Element>(repeating: value, to: shape)
}

//---------------------------------------
// Rank0
@inlinable
public func repeating(_ value: DType) -> Tensor1<DType> {
    repeating(value, Shape1(1), DType.self)
}

@inlinable
public func repeating<Element>(_ value: Element, dtype: Element.Type)
    -> Tensor1<Element> { repeating(value, Shape1(1), dtype) }

//---------------------------------------
// Rank1
@inlinable
public func repeating(
    _ value: DType,
    _ shape: Shape1.Tuple,
    order: StorageOrder = .C
) -> Tensor1<DType> { repeating(value, shape, DType.self, order) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape1.Tuple,
    dtype: Element.Type
) -> Tensor1<Element> { repeating(value, shape, dtype) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape1.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor1<Element> { repeating(value, shape, dtype, order) }

//---------------------------------------
// Rank2
@inlinable
public func repeating(
    _ value: DType,
    _ shape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2<DType> { repeating(value, shape, DType.self, order) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape2.Tuple,
    dtype: Element.Type
) -> Tensor2<Element> { repeating(value, shape, dtype) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape2.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor2<Element> { repeating(value, shape, dtype, order) }

//---------------------------------------
// Rank3
@inlinable
public func repeating(
    _ value: DType,
    _ shape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3<DType> { repeating(value, shape, DType.self, order) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape3.Tuple,
    dtype: Element.Type
) -> Tensor3<Element> { repeating(value, shape, dtype) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape3.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor3<Element> { repeating(value, shape, dtype, order) }

//---------------------------------------
// Rank4
@inlinable
public func repeating(
    _ value: DType,
    _ shape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4<DType> { repeating(value, shape, DType.self, order) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape4.Tuple,
    dtype: Element.Type
) -> Tensor4<Element> { repeating(value, shape, dtype) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape4.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor4<Element> { repeating(value, shape, dtype, order) }

//---------------------------------------
// Rank5
@inlinable
public func repeating(
    _ value: DType,
    _ shape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5<DType> { repeating(value, shape, DType.self, order) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape5.Tuple,
    dtype: Element.Type
) -> Tensor5<Element> { repeating(value, shape, dtype) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape5.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor5<Element> { repeating(value, shape, dtype, order) }

//---------------------------------------
// Rank6
@inlinable
public func repeating(
    _ value: DType,
    _ shape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6<DType> { repeating(value, shape, DType.self, order) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape6.Tuple,
    dtype: Element.Type
) -> Tensor6<Element> { repeating(value, shape, dtype) }

@inlinable
public func repeating<Element>(
    _ value: Element,
    _ shape: Shape6.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor6<Element> { repeating(value, shape, dtype, order) }


//==============================================================================
/// repeating(value:like:
/// Return a new tensor of given shape and type repeating `value`
/// - Parameters:
///  - value: Fill value.
///  - prototype: unspecified attributes are copied from this tensor
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
///  - shape: Int or tuple of Int
///    Shape of the repeated value, e.g., (2, 3) or 2.
/// - Returns: Fill of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.

// same type and shape
@inlinable
public func repeating<T>(
    _ value: T.Element,
    like prototype: T,
    order: StorageOrder? = nil
) -> Tensor<T.Shape, T.Element> where T: TensorType
{
    repeating(value, prototype.shape, T.Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@inlinable public func repeating<T>(
    _ value: T.Element,
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return repeating(value, shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank2
@inlinable public func repeating<T>(
    _ value: T.Element,
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return repeating(value, shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank3
@inlinable public func repeating<T>(
    _ value: T.Element,
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return repeating(value, shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank4
@inlinable public func repeating<T>(
    _ value: T.Element,
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return repeating(value, shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank5
@inlinable public func repeating<T>(
    _ value: T.Element,
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return repeating(value, shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank6
@inlinable public func repeating<T>(
    _ value: T.Element,
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return repeating(value, shape, T.Element.self, order ?? prototype.storageOrder)
}


//------------------------------------------------------------------------------
// different type same shape
@inlinable
public func repeating<T, Element>(
    _ value: Element,
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<T.Shape, Element> where T: TensorType
{
    repeating(value, prototype.shape, Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@inlinable public func repeating<T, Element>(
    _ value: Element,
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return repeating(value, shape, Element.self, order ?? prototype.storageOrder)
}

// Rank2
@inlinable public func repeating<T, Element>(
    _ value: Element,
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return repeating(value, shape, Element.self, order ?? prototype.storageOrder)
}

// Rank3
@inlinable public func repeating<T, Element>(
    _ value: Element,
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return repeating(value, shape, Element.self, order ?? prototype.storageOrder)
}

// Rank4
@inlinable public func repeating<T, Element>(
    _ value: Element,
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return repeating(value, shape, Element.self, order ?? prototype.storageOrder)
}

// Rank5
@inlinable public func repeating<T, Element>(
    _ value: Element,
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return repeating(value, shape, Element.self, order ?? prototype.storageOrder)
}

// Rank6
@inlinable public func repeating<T, Element>(
    _ value: Element,
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return repeating(value, shape, Element.self, order ?? prototype.storageOrder)
}


