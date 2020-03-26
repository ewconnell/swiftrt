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
/// full
/// Return a new tensor of given shape and type, without initializing entries.
/// - Parameters:
///  - shape: Int or tuple of Int
///    Shape of the array, e.g., (2, 3) or 2.
///  - fillValue: Fill value.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
/// - Returns: Fill of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.
@inlinable
public func full<Shape, Element>(
    _ shape: Shape.Tuple,
    _ fillValue: Element,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> FillTensor<Shape, Element> where Shape: Shaped
{
    full(Shape(shape), fillValue, dtype, order)
}

@inlinable
public func full<Shape, Element>(
    _ shape: Shape,
    _ fillValue: Element,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> FillTensor<Shape, Element> where Shape: Shaped
{
    FillTensor(shape, element: fillValue, order: order)
}

//---------------------------------------
// T0
@inlinable
public func full(_ fillValue: DType) -> Fill1<DType> {
    full(Shape1(1), fillValue, DType.self)
}

@inlinable
public func full<Element>(_ fillValue: Element, dtype: Element.Type)
    -> Fill1<Element> { full(Shape1(1), fillValue, dtype) }

//---------------------------------------
// T1
@inlinable
public func full(_ shape: Shape1.Tuple, _ fillValue: DType, order: StorageOrder = .C)
    -> Fill1<DType> { full(shape, fillValue, DType.self, order) }

@inlinable
public func full<Element>(_ shape: Shape1.Tuple, _ fillValue: Element,
                          dtype: Element.Type)
    -> Fill1<Element> { full(shape, fillValue, dtype) }

@inlinable
public func full<Element>(_ shape: Shape1.Tuple, _ fillValue: Element,
                          dtype: Element.Type, order: StorageOrder = .C)
    -> Fill1<Element> { full(shape, fillValue, dtype, order) }

//---------------------------------------
// T2
@inlinable
public func full(_ shape: Shape2.Tuple, _ fillValue: DType, order: StorageOrder = .C)
    -> Fill2<DType> { full(shape, fillValue, DType.self, order) }

@inlinable
public func full<Element>(_ shape: Shape2.Tuple, _ fillValue: Element,
                          dtype: Element.Type)
    -> Fill2<Element> { full(shape, fillValue, dtype) }

@inlinable
public func full<Element>(_ shape: Shape2.Tuple, _ fillValue: Element,
                          dtype: Element.Type, order: StorageOrder = .C)
    -> Fill2<Element> { full(shape, fillValue, dtype, order) }

//---------------------------------------
// T3
@inlinable
public func full(_ shape: Shape3.Tuple, _ fillValue: DType, order: StorageOrder = .C)
    -> Fill3<DType> { full(shape, fillValue, DType.self, order) }

@inlinable
public func full<Element>(_ shape: Shape3.Tuple, _ fillValue: Element,
                          dtype: Element.Type)
    -> Fill3<Element> { full(shape, fillValue, dtype) }

@inlinable
public func full<Element>(_ shape: Shape3.Tuple, _ fillValue: Element,
                          dtype: Element.Type, order: StorageOrder = .C)
    -> Fill3<Element> { full(shape, fillValue, dtype, order) }

//---------------------------------------
// T4
@inlinable
public func full(_ shape: Shape4.Tuple, _ fillValue: DType, order: StorageOrder = .C)
    -> Fill4<DType> { full(shape, fillValue, DType.self, order) }

@inlinable
public func full<Element>(_ shape: Shape4.Tuple, _ fillValue: Element,
                          dtype: Element.Type)
    -> Fill4<Element> { full(shape, fillValue, dtype) }

@inlinable
public func full<Element>(_ shape: Shape4.Tuple, _ fillValue: Element,
                          dtype: Element.Type, order: StorageOrder = .C)
    -> Fill4<Element> { full(shape, fillValue, dtype, order) }

//---------------------------------------
// T5
@inlinable
public func full(_ shape: Shape5.Tuple, _ fillValue: DType, order: StorageOrder = .C)
    -> Fill5<DType> { full(shape, fillValue, DType.self, order) }

@inlinable
public func full<Element>(_ shape: Shape5.Tuple, _ fillValue: Element,
                          dtype: Element.Type)
    -> Fill5<Element> { full(shape, fillValue, dtype) }

@inlinable
public func full<Element>(_ shape: Shape5.Tuple, _ fillValue: Element,
                          dtype: Element.Type, order: StorageOrder = .C)
    -> Fill5<Element> { full(shape, fillValue, dtype, order) }

//==============================================================================
/// full
/// Return a new tensor of given shape and type, without initializing entries.
/// - Parameters:
///  - prototype: unspecified attributes are copied from this tensor
///  - fillValue: Fill value.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
///  - shape: Int or tuple of Int
///    Shape of the full array, e.g., (2, 3) or 2.
/// - Returns: Fill of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.

// same type and shape
@inlinable
public func full<T>(
    like prototype: T,
    _ fillValue: T.Element,
    order: StorageOrder? = nil
) -> FillTensor<T.Shape, T.Element> where T: Tensor
{
    full(prototype.shape, fillValue, T.Element.self, order ?? prototype.order)
}

//---------------------------------------
// same type different shape
// T1
@inlinable public func full<T>(
    like prototype: T, _ fillValue: T.Element,
    order: StorageOrder? = nil, shape: Shape1.Tuple
) -> FillTensor<Shape1, T.Element> where T: Tensor
{
    assert(prototype.count == Shape1(shape).elementCount())
    return full(shape, fillValue, T.Element.self, order ?? prototype.order)
}

// T2
@inlinable public func full<T>(
    like prototype: T, _ fillValue: T.Element,
    order: StorageOrder? = nil, shape: Shape2.Tuple
) -> FillTensor<Shape2, T.Element> where T: Tensor
{
    assert(prototype.count == Shape2(shape).elementCount())
    return full(shape, fillValue, T.Element.self, order ?? prototype.order)
}

// T3
@inlinable public func full<T>(
    like prototype: T, _ fillValue: T.Element,
    order: StorageOrder? = nil, shape: Shape3.Tuple
) -> FillTensor<Shape3, T.Element> where T: Tensor
{
    assert(prototype.count == Shape3(shape).elementCount())
    return full(shape, fillValue, T.Element.self, order ?? prototype.order)
}

// T4
@inlinable public func full<T>(
    like prototype: T, _ fillValue: T.Element,
    order: StorageOrder? = nil, shape: Shape4.Tuple
) -> FillTensor<Shape4, T.Element> where T: Tensor
{
    assert(prototype.count == Shape4(shape).elementCount())
    return full(shape, fillValue, T.Element.self, order ?? prototype.order)
}

// T5
@inlinable public func full<T>(
    like prototype: T, _ fillValue: T.Element,
    order: StorageOrder? = nil, shape: Shape5.Tuple
) -> FillTensor<Shape5, T.Element> where T: Tensor
{
    assert(prototype.count == Shape5(shape).elementCount())
    return full(shape, fillValue, T.Element.self, order ?? prototype.order)
}

//==============================================================================
//---------------------------------------
// different type same shape
@inlinable
public func full<T, Element>(
    like prototype: T,
    _ fillValue: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> FillTensor<T.Shape, Element> where T: Tensor
{
    full(prototype.shape, fillValue, Element.self, order ?? prototype.order)
}

//---------------------------------------
// different type, different shape

// T1
@inlinable public func full<T, Element>(
    like prototype: T,
    _ fillValue: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> FillTensor<Shape1, Element> where T: Tensor
{
    assert(prototype.count == Shape1(shape).elementCount())
    return full(shape, fillValue, Element.self, order ?? prototype.order)
}

// T2
@inlinable public func full<T, Element>(
    like prototype: T,
    _ fillValue: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> FillTensor<Shape2, Element> where T: Tensor
{
    assert(prototype.count == Shape2(shape).elementCount())
    return full(shape, fillValue, Element.self, order ?? prototype.order)
}

// T3
@inlinable public func full<T, Element>(
    like prototype: T,
    _ fillValue: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> FillTensor<Shape3, Element> where T: Tensor
{
    assert(prototype.count == Shape3(shape).elementCount())
    return full(shape, fillValue, Element.self, order ?? prototype.order)
}

// T4
@inlinable public func full<T, Element>(
    like prototype: T,
    _ fillValue: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> FillTensor<Shape4, Element> where T: Tensor
{
    assert(prototype.count == Shape4(shape).elementCount())
    return full(shape, fillValue, Element.self, order ?? prototype.order)
}

// T5
@inlinable public func full<T, Element>(
    like prototype: T,
    _ fillValue: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> FillTensor<Shape5, Element> where T: Tensor
{
    assert(prototype.count == Shape5(shape).elementCount())
    return full(shape, fillValue, Element.self, order ?? prototype.order)
}
