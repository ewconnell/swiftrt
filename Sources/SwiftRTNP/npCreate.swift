//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// numpy array creation routine reference
// https://numpy.org/doc/1.18/reference/routines.array-creation.html
import SwiftRT

//==============================================================================
/// empty
/// Return a new tensor of given shape and type, without initializing entries.
/// - Parameters:
///  - shape: Int or tuple of Int
///    Shape of the empty array, e.g., (2, 3) or 2.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is Float.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
/// - Returns: Tensor of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.
@inlinable
public func empty<Bounds, Element>(
    _ shape: Bounds.Tuple,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Bounds, Element> where Bounds: ShapeBounds
{
    empty(Bounds(shape), dtype, order)
}

@inlinable
public func empty<Bounds, Element>(
    _ shape: Bounds, _ dtype: Element.Type, _ order: StorageOrder = .C
) -> Tensor<Bounds, Element> where Bounds: ShapeBounds
{
    Tensor<Bounds, Element>(bounds: shape, storage: order)
}

//---------------------------------------
// T0
@inlinable
public func empty() -> Tensor1<Float> {
    empty(Bounds1(1), Float.self)
}

@inlinable
public func empty<Element>(dtype: Element.Type)
    -> Tensor1<Element> { empty(Bounds1(1), dtype) }

//---------------------------------------
// T1
@inlinable
public func empty(_ shape: Bounds1.Tuple, order: StorageOrder = .C)
    -> Tensor1<Float> { empty(shape, Float.self, order) }

@inlinable
public func empty<Element>(_ shape: Bounds1.Tuple, dtype: Element.Type)
    -> Tensor1<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Bounds1.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor1<Element> { empty(shape, dtype, order) }

//---------------------------------------
// T2
@inlinable
public func empty(_ shape: Bounds2.Tuple, order: StorageOrder = .C)
    -> Tensor2<Float> { empty(shape, Float.self, order) }

@inlinable
public func empty<Element>(_ shape: Bounds2.Tuple, dtype: Element.Type)
    -> Tensor2<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Bounds2.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor2<Element> { empty(shape, dtype, order) }

//---------------------------------------
// T3
@inlinable
public func empty(_ shape: Bounds3.Tuple, order: StorageOrder = .C)
    -> Tensor3<Float> { empty(shape, Float.self, order) }

@inlinable
public func empty<Element>(_ shape: Bounds3.Tuple, dtype: Element.Type)
    -> Tensor3<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Bounds3.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor3<Element> { empty(shape, dtype, order) }

//---------------------------------------
// T4
@inlinable
public func empty(_ shape: Bounds4.Tuple, order: StorageOrder = .C)
    -> Tensor4<Float> { empty(shape, Float.self, order) }

@inlinable
public func empty<Element>(_ shape: Bounds4.Tuple, dtype: Element.Type)
    -> Tensor4<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Bounds4.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor4<Element> { empty(shape, dtype, order) }

//---------------------------------------
// T5
@inlinable
public func empty(_ shape: Bounds5.Tuple, order: StorageOrder = .C)
    -> Tensor5<Float> { empty(shape, Float.self, order) }

@inlinable
public func empty<Element>(_ shape: Bounds5.Tuple, dtype: Element.Type)
    -> Tensor5<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Bounds5.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor5<Element> { empty(shape, dtype, order) }

//==============================================================================
/// empty_like
/// Return a new tensor of given shape and type, without initializing entries.
/// - Parameters:
///  - prototype: unspecified attributes are copied from this tensor
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is Float.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
///  - shape: Int or tuple of Int
///    Shape of the empty array, e.g., (2, 3) or 2.
/// - Returns: Tensor of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.
//@inlinable
//public func empty_like<T, Bounds, Element>(
//    _ prototype: T,
//    _ dtype: Element.Type,
//    _ order: StorageOrder = .C,
//    _ shape: Bounds.Tuple
//) -> Tensor<Bounds, Element>
//    where T: TensorView, Bounds: ShapeBounds
//{
//    empty(Bounds(shape), dtype, order)
//}

//---------------------------------------
// same type and shape
@inlinable
public func empty_like<T>(
    _ prototype: T,
    order: StorageOrder? = nil
) -> Tensor<T.Bounds, T.Element> where T: TensorView
{
    empty(prototype.bounds, T.Element.self, order ?? prototype.shape.order)
}

//---------------------------------------
// same type different shape
// T1
@inlinable public func empty_like<T>(
    _ prototype: T, order: StorageOrder? = nil, shape bounds: Bounds1.Tuple
) -> Tensor<Bounds1, T.Element> where T: TensorView
{
    assert(prototype.count == Bounds1(bounds).elementCount())
    return empty(bounds, T.Element.self, order ?? prototype.shape.order)
}

// T2
@inlinable public func empty_like<T>(
    _ prototype: T, order: StorageOrder? = nil, shape bounds: Bounds2.Tuple
) -> Tensor<Bounds2, T.Element> where T: TensorView
{
    assert(prototype.count == Bounds2(bounds).elementCount())
    return empty(bounds, T.Element.self, order ?? prototype.shape.order)
}

// T3
@inlinable public func empty_like<T>(
    _ prototype: T, order: StorageOrder? = nil, shape bounds: Bounds3.Tuple
) -> Tensor<Bounds3, T.Element> where T: TensorView
{
    assert(prototype.count == Bounds3(bounds).elementCount())
    return empty(bounds, T.Element.self, order ?? prototype.shape.order)
}

// T4
@inlinable public func empty_like<T>(
    _ prototype: T, order: StorageOrder? = nil, shape bounds: Bounds4.Tuple
) -> Tensor<Bounds4, T.Element> where T: TensorView
{
    assert(prototype.count == Bounds4(bounds).elementCount())
    return empty(bounds, T.Element.self, order ?? prototype.shape.order)
}

// T5
@inlinable public func empty_like<T>(
    _ prototype: T, order: StorageOrder? = nil, shape bounds: Bounds5.Tuple
) -> Tensor<Bounds5, T.Element> where T: TensorView
{
    assert(prototype.count == Bounds5(bounds).elementCount())
    return empty(bounds, T.Element.self, order ?? prototype.shape.order)
}

//==============================================================================
//---------------------------------------
// different type same shape
@inlinable
public func empty_like<T, Element>(
    _ prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<T.Bounds, Element> where T: TensorView
{
    empty(prototype.bounds, Element.self, order ?? prototype.shape.order)
}

//---------------------------------------
// different type, different shape

// T1
@inlinable public func empty_like<T, Element>(
    _ prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape bounds: Bounds1.Tuple
) -> Tensor<Bounds1, Element> where T: TensorView
{
    assert(prototype.count == Bounds1(bounds).elementCount())
    return empty(bounds, Element.self, order ?? prototype.shape.order)
}

// T2
@inlinable public func empty_like<T, Element>(
    _ prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape bounds: Bounds2.Tuple
) -> Tensor<Bounds2, Element> where T: TensorView
{
    assert(prototype.count == Bounds2(bounds).elementCount())
    return empty(bounds, Element.self, order ?? prototype.shape.order)
}

// T3
@inlinable public func empty_like<T, Element>(
    _ prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape bounds: Bounds3.Tuple
) -> Tensor<Bounds3, Element> where T: TensorView
{
    assert(prototype.count == Bounds3(bounds).elementCount())
    return empty(bounds, Element.self, order ?? prototype.shape.order)
}

// T4
@inlinable public func empty_like<T, Element>(
    _ prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape bounds: Bounds4.Tuple
) -> Tensor<Bounds4, Element> where T: TensorView
{
    assert(prototype.count == Bounds4(bounds).elementCount())
    return empty(bounds, Element.self, order ?? prototype.shape.order)
}

// T5
@inlinable public func empty_like<T, Element>(
    _ prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape bounds: Bounds5.Tuple
) -> Tensor<Bounds5, Element> where T: TensorView
{
    assert(prototype.count == Bounds5(bounds).elementCount())
    return empty(bounds, Element.self, order ?? prototype.shape.order)
}
