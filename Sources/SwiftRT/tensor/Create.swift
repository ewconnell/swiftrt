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

//public typealias Tensor1<Map, Element> = Tensor<Shape1, Map, Element>
//    where Map: ElementMap
//
////==============================================================================
///// empty
///// Return a new tensor of given shape and type, without initializing entries.
///// - Parameters:
/////  - shape: Int or tuple of Int
/////    Shape of the empty array, e.g., (2, 3) or 2.
/////  - dtype: data-type, optional
/////    Desired output data-type for the array, e.g, Int8. Default is Float.
/////  - order: { .C, .F }, optional, default .C
/////    Whether to store multi-dimensional data in row-major (C-style)
/////    or column-major (Fortran-style) order in memory.
///// - Returns: Tensor of uninitialized (arbitrary) data of the given shape,
/////   dtype, and order. Elements will not be initialized.
//@inlinable
//public func empty<Shape, Map, Element>(
//    _ shape: Shape.Tuple,
//    _ dtype: Element.Type,
//    _ map: Map
//) -> Tensor<Shape, Map, Element> where Shape: Shaped
//{
//    empty(Shape(shape), dtype, map)
//}
//
@inlinable
public func empty<Shape, Element>(
    _ shape: Shape,
    _ dtype: Element.Type
) -> DenseTensor<Shape, Element> where Shape: Shaped
{
    DenseTensor(shape)
}

////---------------------------------------
//// T0
//@inlinable
//public func empty() -> Tensor1<DenseRowMap, Float> {
//    empty(Shape1(1), Float.self, )
//}

//@inlinable
//public func empty<Element>(dtype: Element.Type)
//    -> Tensor1<Element> { empty(Bounds1(1), dtype) }
//
////---------------------------------------
//// T1
//@inlinable
//public func empty(_ shape: Bounds1.Tuple, order: StorageOrder = .C)
//    -> Tensor1<Float> { empty(shape, Float.self, order) }
//
//@inlinable
//public func empty<Element>(_ shape: Bounds1.Tuple, dtype: Element.Type)
//    -> Tensor1<Element> { empty(shape, dtype) }
//
//@inlinable
//public func empty<Element>(_ shape: Bounds1.Tuple, dtype: Element.Type,
//                           order: StorageOrder = .C)
//    -> Tensor1<Element> { empty(shape, dtype, order) }
//
