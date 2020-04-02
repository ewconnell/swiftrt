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
// DO NOT EDIT. THIS FILE IS GENERATED FROM array.swift.gyb
//
//******************************************************************************

//==============================================================================
/// array
/// Return a new tensor of given shape and type, without initializing entries.
/// - Parameters:
///  - elements: a collection of elements used to initialize storage
///  - shape: Int or tuple of Int describing the dimensions of the array
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
/// - Returns: Dense of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.

//==============================================================================
// Rank1 from a flat collection
// same type
@inlinable public func array<C>(flat elements: C)
    -> Dense1<C.Element> where C: Collection
{
    Dense1(elements, Shape1(elements.count))
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C>(flat elements: C)
    -> Dense1<DType> where C: Collection, C.Element: BinaryInteger
{
    Dense1(elements, Shape1(elements.count))
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C, Element>(
    flat elements: C,
    dtype: Element.Type
) -> Dense1<Element>
    where C: Collection, C.Element: BinaryInteger, Element: Numeric
{
    Dense1<Element>(elements, Shape1(elements.count))
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<C, Element>(
    flat elements: C,
    dtype: Element.Type
) -> Dense1<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryInteger
{
    Dense1<Element>(elements, Shape1(elements.count))
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<C, Element>(
    flat elements: C,
    dtype: Element.Type
) -> Dense1<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    Dense1<Element>(elements, Shape1(elements.count))
}

//------------------------------------------------------------------------------
// Rank2 array from a flat collection
// same type
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Dense2<C.Element> where C: Collection
{
    Dense2(elements, Shape2(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Dense2<DType> where C: Collection, C.Element: BinaryInteger
{
    Dense2(elements, Shape2(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape2.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense2<Element>
    where C: Collection, C.Element: BinaryInteger, Element: Numeric
{
    Dense2<Element>(elements, Shape2(shape), order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape2.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense2<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryInteger
{
    Dense2<Element>(elements, Shape2(shape), order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape2.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense2<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    Dense2<Element>(elements, Shape2(shape), order: order)
}

//------------------------------------------------------------------------------
// Rank3 array from a flat collection
// same type
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Dense3<C.Element> where C: Collection
{
    Dense3(elements, Shape3(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Dense3<DType> where C: Collection, C.Element: BinaryInteger
{
    Dense3(elements, Shape3(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape3.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense3<Element>
    where C: Collection, C.Element: BinaryInteger, Element: Numeric
{
    Dense3<Element>(elements, Shape3(shape), order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape3.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense3<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryInteger
{
    Dense3<Element>(elements, Shape3(shape), order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape3.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense3<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    Dense3<Element>(elements, Shape3(shape), order: order)
}

//------------------------------------------------------------------------------
// Rank4 array from a flat collection
// same type
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Dense4<C.Element> where C: Collection
{
    Dense4(elements, Shape4(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Dense4<DType> where C: Collection, C.Element: BinaryInteger
{
    Dense4(elements, Shape4(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape4.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense4<Element>
    where C: Collection, C.Element: BinaryInteger, Element: Numeric
{
    Dense4<Element>(elements, Shape4(shape), order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape4.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense4<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryInteger
{
    Dense4<Element>(elements, Shape4(shape), order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape4.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense4<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    Dense4<Element>(elements, Shape4(shape), order: order)
}

//------------------------------------------------------------------------------
// Rank5 array from a flat collection
// same type
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Dense5<C.Element> where C: Collection
{
    Dense5(elements, Shape5(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Dense5<DType> where C: Collection, C.Element: BinaryInteger
{
    Dense5(elements, Shape5(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape5.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense5<Element>
    where C: Collection, C.Element: BinaryInteger, Element: Numeric
{
    Dense5<Element>(elements, Shape5(shape), order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape5.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense5<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryInteger
{
    Dense5<Element>(elements, Shape5(shape), order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape5.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense5<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    Dense5<Element>(elements, Shape5(shape), order: order)
}

//------------------------------------------------------------------------------
// Rank6 array from a flat collection
// same type
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Dense6<C.Element> where C: Collection
{
    Dense6(elements, Shape6(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C>(
    flat elements: C,
    _ shape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Dense6<DType> where C: Collection, C.Element: BinaryInteger
{
    Dense6(elements, Shape6(shape), order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape6.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense6<Element>
    where C: Collection, C.Element: BinaryInteger, Element: Numeric
{
    Dense6<Element>(elements, Shape6(shape), order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape6.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense6<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryInteger
{
    Dense6<Element>(elements, Shape6(shape), order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<C, Element>(
    flat elements: C,
    _ shape: Shape6.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense6<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    Dense6<Element>(elements, Shape6(shape), order: order)
}


//==============================================================================
// Rank1 from 1D Swift array
// same type
@inlinable public func array<Element>(
    _ elements: [Element],
    order: StorageOrder = .C
) -> Dense1<Element>
{
    let shape = Shape1(
        elements.count)

    let flatElements = elements
    return Dense1<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement>(
    _ elements: [ArrayElement],
    order: StorageOrder = .C
) -> Dense1<DType> where ArrayElement: BinaryInteger
{
    let shape = Shape1(
        elements.count)

    let flatElements = elements
    return Dense1<DType>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement, Element>(
    _ elements: [ArrayElement],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense1<Element>
    where ArrayElement: BinaryInteger, Element: Numeric
{
    let shape = Shape1(
        elements.count)

    let flatElements = elements
    return Dense1<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<ArrayElement, Element>(
    _ elements: [ArrayElement],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense1<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryInteger
{
    let shape = Shape1(
        elements.count)

    let flatElements = elements
    return Dense1<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<ArrayElement, Element>(
    _ elements: [ArrayElement],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense1<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    let shape = Shape1(
        elements.count)

    let flatElements = elements
    return Dense1<Element>(flatElements, shape, order: order)
}

//==============================================================================
// Rank2 from 2D Swift array
// same type
@inlinable public func array<Element>(
    _ elements: [[Element]],
    order: StorageOrder = .C
) -> Dense2<Element>
{
    let shape = Shape2(
        elements.count,
        elements.first!.count)

    let flatElements = elements.joined()
    return Dense2<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement>(
    _ elements: [[ArrayElement]],
    order: StorageOrder = .C
) -> Dense2<DType> where ArrayElement: BinaryInteger
{
    let shape = Shape2(
        elements.count,
        elements.first!.count)

    let flatElements = elements.joined()
    return Dense2<DType>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[ArrayElement]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense2<Element>
    where ArrayElement: BinaryInteger, Element: Numeric
{
    let shape = Shape2(
        elements.count,
        elements.first!.count)

    let flatElements = elements.joined()
    return Dense2<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[ArrayElement]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense2<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryInteger
{
    let shape = Shape2(
        elements.count,
        elements.first!.count)

    let flatElements = elements.joined()
    return Dense2<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[ArrayElement]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense2<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    let shape = Shape2(
        elements.count,
        elements.first!.count)

    let flatElements = elements.joined()
    return Dense2<Element>(flatElements, shape, order: order)
}

//==============================================================================
// Rank3 from 3D Swift array
// same type
@inlinable public func array<Element>(
    _ elements: [[[Element]]],
    order: StorageOrder = .C
) -> Dense3<Element>
{
    let shape = Shape3(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count)

    let flatElements = elements.joined().joined()
    return Dense3<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement>(
    _ elements: [[[ArrayElement]]],
    order: StorageOrder = .C
) -> Dense3<DType> where ArrayElement: BinaryInteger
{
    let shape = Shape3(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count)

    let flatElements = elements.joined().joined()
    return Dense3<DType>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[ArrayElement]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense3<Element>
    where ArrayElement: BinaryInteger, Element: Numeric
{
    let shape = Shape3(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count)

    let flatElements = elements.joined().joined()
    return Dense3<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[ArrayElement]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense3<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryInteger
{
    let shape = Shape3(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count)

    let flatElements = elements.joined().joined()
    return Dense3<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[ArrayElement]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense3<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    let shape = Shape3(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count)

    let flatElements = elements.joined().joined()
    return Dense3<Element>(flatElements, shape, order: order)
}

//==============================================================================
// Rank4 from 4D Swift array
// same type
@inlinable public func array<Element>(
    _ elements: [[[[Element]]]],
    order: StorageOrder = .C
) -> Dense4<Element>
{
    let shape = Shape4(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined()
    return Dense4<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement>(
    _ elements: [[[[ArrayElement]]]],
    order: StorageOrder = .C
) -> Dense4<DType> where ArrayElement: BinaryInteger
{
    let shape = Shape4(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined()
    return Dense4<DType>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[[ArrayElement]]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense4<Element>
    where ArrayElement: BinaryInteger, Element: Numeric
{
    let shape = Shape4(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined()
    return Dense4<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[[ArrayElement]]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense4<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryInteger
{
    let shape = Shape4(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined()
    return Dense4<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[[ArrayElement]]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense4<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    let shape = Shape4(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined()
    return Dense4<Element>(flatElements, shape, order: order)
}

//==============================================================================
// Rank5 from 5D Swift array
// same type
@inlinable public func array<Element>(
    _ elements: [[[[[Element]]]]],
    order: StorageOrder = .C
) -> Dense5<Element>
{
    let shape = Shape5(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined()
    return Dense5<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement>(
    _ elements: [[[[[ArrayElement]]]]],
    order: StorageOrder = .C
) -> Dense5<DType> where ArrayElement: BinaryInteger
{
    let shape = Shape5(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined()
    return Dense5<DType>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[[[ArrayElement]]]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense5<Element>
    where ArrayElement: BinaryInteger, Element: Numeric
{
    let shape = Shape5(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined()
    return Dense5<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[[[ArrayElement]]]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense5<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryInteger
{
    let shape = Shape5(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined()
    return Dense5<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[[[ArrayElement]]]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense5<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    let shape = Shape5(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined()
    return Dense5<Element>(flatElements, shape, order: order)
}

//==============================================================================
// Rank6 from 6D Swift array
// same type
@inlinable public func array<Element>(
    _ elements: [[[[[[Element]]]]]],
    order: StorageOrder = .C
) -> Dense6<Element>
{
    let shape = Shape6(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count,
        elements.first!.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined().joined()
    return Dense6<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement>(
    _ elements: [[[[[[ArrayElement]]]]]],
    order: StorageOrder = .C
) -> Dense6<DType> where ArrayElement: BinaryInteger
{
    let shape = Shape6(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count,
        elements.first!.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined().joined()
    return Dense6<DType>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[[[[ArrayElement]]]]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense6<Element>
    where ArrayElement: BinaryInteger, Element: Numeric
{
    let shape = Shape6(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count,
        elements.first!.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined().joined()
    return Dense6<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element integer
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[[[[ArrayElement]]]]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense6<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryInteger
{
    let shape = Shape6(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count,
        elements.first!.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined().joined()
    return Dense6<Element>(flatElements, shape, order: order)
}

/// implicitly casts from C.Element float -> Element float
@inlinable public func array<ArrayElement, Element>(
    _ elements: [[[[[[ArrayElement]]]]]],
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Dense6<Element>
    where ArrayElement: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    let shape = Shape6(
        elements.count,
        elements.first!.count,
        elements.first!.first!.count,
        elements.first!.first!.first!.count,
        elements.first!.first!.first!.first!.count,
        elements.first!.first!.first!.first!.first!.count)

    let flatElements = elements.joined().joined().joined().joined().joined()
    return Dense6<Element>(flatElements, shape, order: order)
}

