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
//@inlinable
//public func array<C, Shape, Element>(
//    _ elements: C,
//    _ shape: Shape.Tuple,
//    _ dtype: Element.Type,
//    _ order: StorageOrder = .C
//) -> DenseTensor<Shape, Element>
//    where Shape: TensorShape, C: Collection
//{
//    array(elements, Shape(shape), dtype, order)
//}
//
//@inlinable
//public func array<C, Shape, Element>(
//    _ elements: C,
//    _ shape: Shape,
//    _ dtype: Element.Type,
//    _ order: StorageOrder = .C
//) -> DenseTensor<Shape, Element>
//    where Shape: TensorShape, C: Collection
//{
//    DenseTensor<Shape, Element>(shape, order: order, elements: elements)
//}

//---------------------------------------
// Rank1

// same type
@inlinable public func array<C>(_ elements: C)
    -> Dense1<C.Element> where C: Collection
{
    Dense1(elements, Shape1(elements.count))
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C>(_ elements: C)
    -> Dense1<DType> where C: Collection, C.Element: BinaryInteger
{
    Dense1(elements, Shape1(elements.count))
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C, Element>(
    _ elements: C,
    dtype: Element.Type
) -> Dense1<Element>
    where C: Collection, C.Element: BinaryInteger, Element: Numeric
{
    Dense1<Element>(elements, Shape1(elements.count))
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C, Element>(
    _ elements: C,
    dtype: Element.Type
) -> Dense1<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryInteger
{
    Dense1<Element>(elements, Shape1(elements.count))
}

/// implicitly casts from C.Element integer -> Element
@inlinable public func array<C, Element>(
    _ elements: C,
    dtype: Element.Type
) -> Dense1<Element>
    where C: Collection, C.Element: BinaryFloatingPoint, Element: BinaryFloatingPoint
{
    Dense1<Element>(elements, Shape1(elements.count))
}
