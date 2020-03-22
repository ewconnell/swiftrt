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

//==============================================================================
/// DType
/// the implicit tensor Element type
public typealias DType = Float

//==============================================================================
///
public typealias Tensor0<Element> = Tensor<Bounds1, Element>
public typealias Tensor1<Element> = Tensor<Bounds1, Element>
public typealias Tensor2<Element> = Tensor<Bounds2, Element>
public typealias Tensor3<Element> = Tensor<Bounds3, Element>
public typealias Tensor4<Element> = Tensor<Bounds4, Element>
public typealias Tensor5<Element> = Tensor<Bounds5, Element>

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
    Tensor<Bounds, Element>(bounds: shape)
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
