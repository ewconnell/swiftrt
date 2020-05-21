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
/// DType
/// the implicit tensor Element type
public typealias DType = Float

public typealias Tensor1 = Tensor<Shape1,DType>
public typealias Tensor2 = Tensor<Shape2,DType>
public typealias Tensor3 = Tensor<Shape3,DType>
public typealias Tensor4 = Tensor<Shape4,DType>
public typealias Tensor5 = Tensor<Shape5,DType>
public typealias Tensor6 = Tensor<Shape6,DType>

//==============================================================================
/// empty
/// Return a new tensor of given shape and type, without initializing entries.
/// - Parameters:
///  - shape: Int or tuple of Int
///    Shape of the empty array, e.g., (2, 3) or 2.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
/// - Returns: Tensor of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.
@inlinable public func empty<Shape, Element>(
    _ shape: Shape.Tuple,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element: StorageElement
{
    empty(Shape(shape), dtype, order)
}

@inlinable public func empty<Shape, Element>(
    _ shape: Shape,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element: StorageElement
{
    Tensor(shape, order: order)
}

//---------------------------------------
// Rank 0
@inlinable public func empty() -> Tensor<Shape1, DType> {
    empty(Shape1(1), DType.self)
}

@inlinable public func empty<Element>(
    dtype: Element.Type
) -> Tensor<Shape1, Element>
    where Element: StorageElement
{
    empty(Shape1(1), dtype)
}

//---------------------------------------
// Rank1
@inlinable public func empty(_ shape: Shape1.Tuple, order: StorageOrder = .C)
    -> Tensor<Shape1, DType> { empty(shape, DType.self, order) }

@inlinable public func empty<Element>(
    _ shape: Shape1.Tuple,
    dtype: Element.Type
) -> Tensor<Shape1, Element>
    where Element: StorageElement
{
    empty(shape, dtype)
}

@inlinable public func empty<Element>(
    _ shape: Shape1.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape1, Element>
    where Element: StorageElement
{
    empty(shape, dtype, order)
}
    
//---------------------------------------
// Rank2
@inlinable public func empty(_ shape: Shape2.Tuple, order: StorageOrder = .C)
    -> Tensor<Shape2, DType> { empty(shape, DType.self, order) }

@inlinable public func empty<Element>(
    _ shape: Shape2.Tuple,
    dtype: Element.Type
) -> Tensor<Shape2, Element>
    where Element: StorageElement
{
    empty(shape, dtype)
}

@inlinable public func empty<Element>(
    _ shape: Shape2.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape2, Element>
    where Element: StorageElement
{
    empty(shape, dtype, order)
}
    
//---------------------------------------
// Rank3
@inlinable public func empty(_ shape: Shape3.Tuple, order: StorageOrder = .C)
    -> Tensor<Shape3, DType> { empty(shape, DType.self, order) }

@inlinable public func empty<Element>(
    _ shape: Shape3.Tuple,
    dtype: Element.Type
) -> Tensor<Shape3, Element>
    where Element: StorageElement
{
    empty(shape, dtype)
}

@inlinable public func empty<Element>(
    _ shape: Shape3.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape3, Element>
    where Element: StorageElement
{
    empty(shape, dtype, order)
}
    
//---------------------------------------
// Rank4
@inlinable public func empty(_ shape: Shape4.Tuple, order: StorageOrder = .C)
    -> Tensor<Shape4, DType> { empty(shape, DType.self, order) }

@inlinable public func empty<Element>(
    _ shape: Shape4.Tuple,
    dtype: Element.Type
) -> Tensor<Shape4, Element>
    where Element: StorageElement
{
    empty(shape, dtype)
}

@inlinable public func empty<Element>(
    _ shape: Shape4.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape4, Element>
    where Element: StorageElement
{
    empty(shape, dtype, order)
}
    
//---------------------------------------
// Rank5
@inlinable public func empty(_ shape: Shape5.Tuple, order: StorageOrder = .C)
    -> Tensor<Shape5, DType> { empty(shape, DType.self, order) }

@inlinable public func empty<Element>(
    _ shape: Shape5.Tuple,
    dtype: Element.Type
) -> Tensor<Shape5, Element>
    where Element: StorageElement
{
    empty(shape, dtype)
}

@inlinable public func empty<Element>(
    _ shape: Shape5.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape5, Element>
    where Element: StorageElement
{
    empty(shape, dtype, order)
}
    
//---------------------------------------
// Rank6
@inlinable public func empty(_ shape: Shape6.Tuple, order: StorageOrder = .C)
    -> Tensor<Shape6, DType> { empty(shape, DType.self, order) }

@inlinable public func empty<Element>(
    _ shape: Shape6.Tuple,
    dtype: Element.Type
) -> Tensor<Shape6, Element>
    where Element: StorageElement
{
    empty(shape, dtype)
}

@inlinable public func empty<Element>(
    _ shape: Shape6.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape6, Element>
    where Element: StorageElement
{
    empty(shape, dtype, order)
}
    

//==============================================================================
/// empty(like:
/// Return a new tensor of given shape and type, without initializing entries.
/// - Parameters:
///  - prototype: unspecified attributes are copied from this tensor
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
///  - shape: Int or tuple of Int
///    Shape of the empty array, e.g., (2, 3) or 2.
/// - Returns: Tensor of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.

// same type and shape
@inlinable public func empty<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil
) -> Tensor<S,E> where E: StorageElement
{
    empty(prototype.shape, E.self, order ?? prototype.storage.order)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@inlinable public func empty<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, E>
    where E: StorageElement
{
    assert(prototype.count == Shape1(shape).elementCount())
    return empty(shape, E.self, order ?? prototype.storage.order)
}
// Rank2
@inlinable public func empty<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, E>
    where E: StorageElement
{
    assert(prototype.count == Shape2(shape).elementCount())
    return empty(shape, E.self, order ?? prototype.storage.order)
}
// Rank3
@inlinable public func empty<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, E>
    where E: StorageElement
{
    assert(prototype.count == Shape3(shape).elementCount())
    return empty(shape, E.self, order ?? prototype.storage.order)
}
// Rank4
@inlinable public func empty<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, E>
    where E: StorageElement
{
    assert(prototype.count == Shape4(shape).elementCount())
    return empty(shape, E.self, order ?? prototype.storage.order)
}
// Rank5
@inlinable public func empty<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, E>
    where E: StorageElement
{
    assert(prototype.count == Shape5(shape).elementCount())
    return empty(shape, E.self, order ?? prototype.storage.order)
}
// Rank6
@inlinable public func empty<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, E>
    where E: StorageElement
{
    assert(prototype.count == Shape6(shape).elementCount())
    return empty(shape, E.self, order ?? prototype.storage.order)
}

//------------------------------------------------------------------------------
// different type same shape
@inlinable public func empty<S,E,Element>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<S, Element>
    where Element: StorageElement
{
    empty(prototype.shape, Element.self, order ?? prototype.storage.order)
}

//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@inlinable public func empty<S,E,Element>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element>
    where Element: StorageElement
{
    assert(prototype.count == Shape1(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storage.order)
}
// Rank2
@inlinable public func empty<S,E,Element>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element>
    where Element: StorageElement
{
    assert(prototype.count == Shape2(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storage.order)
}
// Rank3
@inlinable public func empty<S,E,Element>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element>
    where Element: StorageElement
{
    assert(prototype.count == Shape3(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storage.order)
}
// Rank4
@inlinable public func empty<S,E,Element>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element>
    where Element: StorageElement
{
    assert(prototype.count == Shape4(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storage.order)
}
// Rank5
@inlinable public func empty<S,E,Element>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element>
    where Element: StorageElement
{
    assert(prototype.count == Shape5(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storage.order)
}
// Rank6
@inlinable public func empty<S,E,Element>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element>
    where Element: StorageElement
{
    assert(prototype.count == Shape6(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storage.order)
}

//==============================================================================
/// full
/// Return a new tensor of given shape and type filled with `value`
/// - Parameters:
///  - shape: Int or tuple of Int
///    Shape of the array, e.g., (2, 3) or 2.
///  - value: Fill value.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
/// - Returns: Fill of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.
@inlinable public func full<Shape, Element>(
    _ shape: Shape.Tuple,
    _ value: Element,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element: StorageElement
{
    full(Shape(shape), value, dtype, order)
}

@inlinable public func full<Shape, Element>(
    _ shape: Shape,
    _ value: Element,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element: StorageElement
{
    var tensor = Tensor<Shape, Element>(shape, order: order)
    fill(&tensor, with: value)
    return tensor
}

//---------------------------------------
// Rank0
@inlinable
public func full(_ value: DType) -> Tensor<Shape1, DType> {
    full(Shape1(1), value, DType.self)
}

@inlinable public func full<Element: StorageElement>(
    _ value: Element, dtype: Element.Type
) -> Tensor<Shape1, Element> {
    full(Shape1(1), value, dtype)
}

//---------------------------------------
// Rank1
@inlinable public func full(
    _ shape: Shape1.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor1 { full(shape, value, DType.self, order) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape1.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor<Shape1, Element> { full(shape, value, dtype) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape1.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape1, Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank2
@inlinable public func full(
    _ shape: Shape2.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor2 { full(shape, value, DType.self, order) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape2.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor<Shape2, Element> { full(shape, value, dtype) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape2.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape2, Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank3
@inlinable public func full(
    _ shape: Shape3.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor3 { full(shape, value, DType.self, order) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape3.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor<Shape3, Element> { full(shape, value, dtype) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape3.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape3, Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank4
@inlinable public func full(
    _ shape: Shape4.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor4 { full(shape, value, DType.self, order) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape4.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor<Shape4, Element> { full(shape, value, dtype) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape4.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape4, Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank5
@inlinable public func full(
    _ shape: Shape5.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor5 { full(shape, value, DType.self, order) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape5.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor<Shape5, Element> { full(shape, value, dtype) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape5.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape5, Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank6
@inlinable public func full(
    _ shape: Shape6.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor6 { full(shape, value, DType.self, order) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape6.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor<Shape6, Element> { full(shape, value, dtype) }

@inlinable public func full<Element: StorageElement>(
    _ shape: Shape6.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape6, Element> { full(shape, value, dtype, order) }


//==============================================================================
/// full(like:
/// Return a new tensor of given shape and type filled with `value`
/// - Parameters:
///  - prototype: unspecified attributes are copied from this tensor
///  - value: Fill value.
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
@inlinable public func full<S,E: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: E,
    order: StorageOrder? = nil
) -> Tensor<S,E>
{
    full(prototype.shape, value, E.self, order ?? prototype.storage.order)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@inlinable public func full<S,E: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: E,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, E>
{
    assert(prototype.count == Shape1(shape).elementCount())
    return full(shape, value, E.self, order ?? prototype.storage.order)
}

// Rank2
@inlinable public func full<S,E: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: E,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, E>
{
    assert(prototype.count == Shape2(shape).elementCount())
    return full(shape, value, E.self, order ?? prototype.storage.order)
}

// Rank3
@inlinable public func full<S,E: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: E,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, E>
{
    assert(prototype.count == Shape3(shape).elementCount())
    return full(shape, value, E.self, order ?? prototype.storage.order)
}

// Rank4
@inlinable public func full<S,E: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: E,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, E>
{
    assert(prototype.count == Shape4(shape).elementCount())
    return full(shape, value, E.self, order ?? prototype.storage.order)
}

// Rank5
@inlinable public func full<S,E: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: E,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, E>
{
    assert(prototype.count == Shape5(shape).elementCount())
    return full(shape, value, E.self, order ?? prototype.storage.order)
}

// Rank6
@inlinable public func full<S,E: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: E,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, E>
{
    assert(prototype.count == Shape6(shape).elementCount())
    return full(shape, value, E.self, order ?? prototype.storage.order)
}


//------------------------------------------------------------------------------
// different type same shape
@inlinable public func full<S,E,Element: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<S, Element>
{
    full(prototype.shape, value, Element.self, order ?? prototype.storage.order)
}

//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@inlinable public func full<S,E,Element: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element>
{
    assert(prototype.count == Shape1(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storage.order)
}

// Rank2
@inlinable public func full<S,E,Element: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element>
{
    assert(prototype.count == Shape2(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storage.order)
}

// Rank3
@inlinable public func full<S,E,Element: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element>
{
    assert(prototype.count == Shape3(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storage.order)
}

// Rank4
@inlinable public func full<S,E,Element: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element>
{
    assert(prototype.count == Shape4(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storage.order)
}

// Rank5
@inlinable public func full<S,E,Element: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element>
{
    assert(prototype.count == Shape5(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storage.order)
}

// Rank6
@inlinable public func full<S,E,Element: StorageElement>(
    like prototype: Tensor<S,E>,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element>
{
    assert(prototype.count == Shape6(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storage.order)
}


//==============================================================================
/// ones
/// Return a new tensor of given shape and type filled with ones
/// - Parameters:
///  - shape: Int or tuple of Int
///    Shape of the array, e.g., (2, 3) or 2.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
/// - Returns: Fill of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.
@inlinable public func ones<Shape, Element: StorageElement>(
    _ shape: Shape.Tuple,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element.Value: Numeric
{
    ones(Shape(shape), dtype, order)
}

@inlinable public func ones<Shape, Element: StorageElement>(
    _ shape: Shape,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element.Value: Numeric
{
    Tensor<Shape, Element>(ones: shape, order: order)
}

//---------------------------------------
// Rank0
@inlinable public func ones() -> Tensor<Shape1, DType> {
    ones(Shape1(1), DType.self)
}

@inlinable public func ones<Element: StorageElement>(
    dtype: Element.Type
) -> Tensor<Shape1, Element>
    where Element.Value: Numeric
{
    ones(Shape1(1), dtype)
}

//---------------------------------------
// Rank1
@inlinable public func ones(
    _ shape: Shape1.Tuple,
    order: StorageOrder = .C
) -> Tensor1 { ones(shape, DType.self, order) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape1.Tuple,
     dtype: Element.Type
) -> Tensor<Shape1, Element>
    where Element.Value: Numeric { ones(shape, dtype) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape1.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape1, Element>
    where Element.Value: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank2
@inlinable public func ones(
    _ shape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2 { ones(shape, DType.self, order) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape2.Tuple,
     dtype: Element.Type
) -> Tensor<Shape2, Element>
    where Element.Value: Numeric { ones(shape, dtype) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape2.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape2, Element>
    where Element.Value: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank3
@inlinable public func ones(
    _ shape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3 { ones(shape, DType.self, order) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape3.Tuple,
     dtype: Element.Type
) -> Tensor<Shape3, Element>
    where Element.Value: Numeric { ones(shape, dtype) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape3.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape3, Element>
    where Element.Value: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank4
@inlinable public func ones(
    _ shape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4 { ones(shape, DType.self, order) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape4.Tuple,
     dtype: Element.Type
) -> Tensor<Shape4, Element>
    where Element.Value: Numeric { ones(shape, dtype) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape4.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape4, Element>
    where Element.Value: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank5
@inlinable public func ones(
    _ shape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5 { ones(shape, DType.self, order) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape5.Tuple,
     dtype: Element.Type
) -> Tensor<Shape5, Element>
    where Element.Value: Numeric { ones(shape, dtype) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape5.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape5, Element>
    where Element.Value: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank6
@inlinable public func ones(
    _ shape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6 { ones(shape, DType.self, order) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape6.Tuple,
     dtype: Element.Type
) -> Tensor<Shape6, Element>
    where Element.Value: Numeric { ones(shape, dtype) }

@inlinable public func ones<Element: StorageElement>(
    _ shape: Shape6.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape6, Element>
    where Element.Value: Numeric { ones(shape, dtype, order) }


//==============================================================================
/// ones(like:
/// Return a new tensor of given shape and type filled with `value`
/// - Parameters:
///  - prototype: unspecified attributes are copied from this tensor
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
///  - shape: Int or tuple of Int
///    Shape of the ones array, e.g., (2, 3) or 2.
/// - Returns: Fill of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.

// same type and shape
@inlinable public func ones<S, E: Numeric>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil
) -> Tensor<S,E> {
    ones(prototype.shape, E.self, order ?? prototype.storage.order)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@inlinable public func ones<S, E: Numeric>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, E> {
    assert(prototype.count == Shape1(shape).elementCount())
    return ones(shape, E.self, order ?? prototype.storage.order)
}

// Rank2
@inlinable public func ones<S, E: Numeric>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, E> {
    assert(prototype.count == Shape2(shape).elementCount())
    return ones(shape, E.self, order ?? prototype.storage.order)
}

// Rank3
@inlinable public func ones<S, E: Numeric>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, E> {
    assert(prototype.count == Shape3(shape).elementCount())
    return ones(shape, E.self, order ?? prototype.storage.order)
}

// Rank4
@inlinable public func ones<S, E: Numeric>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, E> {
    assert(prototype.count == Shape4(shape).elementCount())
    return ones(shape, E.self, order ?? prototype.storage.order)
}

// Rank5
@inlinable public func ones<S, E: Numeric>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, E> {
    assert(prototype.count == Shape5(shape).elementCount())
    return ones(shape, E.self, order ?? prototype.storage.order)
}

// Rank6
@inlinable public func ones<S, E: Numeric>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, E> {
    assert(prototype.count == Shape6(shape).elementCount())
    return ones(shape, E.self, order ?? prototype.storage.order)
}


//------------------------------------------------------------------------------
// different type same shape
@inlinable public func ones<S,E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<S, Element> {
    ones(prototype.shape, Element.self, order ?? prototype.storage.order)
}

//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@inlinable public func ones<S, E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element>
{
    assert(prototype.count == Shape1(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storage.order)
}

// Rank2
@inlinable public func ones<S, E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element>
{
    assert(prototype.count == Shape2(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storage.order)
}

// Rank3
@inlinable public func ones<S, E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element>
{
    assert(prototype.count == Shape3(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storage.order)
}

// Rank4
@inlinable public func ones<S, E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element>
{
    assert(prototype.count == Shape4(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storage.order)
}

// Rank5
@inlinable public func ones<S, E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element>
{
    assert(prototype.count == Shape5(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storage.order)
}

// Rank6
@inlinable public func ones<S, E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element>
{
    assert(prototype.count == Shape6(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storage.order)
}


//==============================================================================
/// zeros
/// Return a new tensor of given shape and type filled with zeros
/// - Parameters:
///  - shape: Int or tuple of Int
///    Shape of the array, e.g., (2, 3) or 2.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
/// - Returns: Fill of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.
@inlinable public func zeros<S,E>(
    _ shape: S.Tuple,
    _ dtype: E.Type,
    _ order: StorageOrder = .C
) -> Tensor<S,E>
    where E: StorageElement, E.Value: Numeric
{
    zeros(S(shape), dtype, order)
}

@inlinable public func zeros<S,E>(
    _ shape: S,
    _ dtype: E.Type,
    _ order: StorageOrder = .C
) -> Tensor<S,E>
    where E: StorageElement, E.Value: Numeric
{
    Tensor<S,E>(zeros: shape, order: order)
}

//---------------------------------------
// Rank0
@inlinable public func zeros() -> Tensor<Shape1, DType> {
    zeros(Shape1(1), DType.self)
}

@inlinable public func zeros<Element>(
    dtype: Element.Type
) -> Tensor<Shape1, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(Shape1(1), dtype)
}

//---------------------------------------
// Rank1
@inlinable public func zeros(
    _ shape: Shape1.Tuple,
    order: StorageOrder = .C
) -> Tensor1 { zeros(shape, DType.self, order) }

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape1.Tuple,
    dtype: Element.Type
) -> Tensor<Shape1, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype)
}

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape1.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape1, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype, order)
}

//---------------------------------------
// Rank2
@inlinable public func zeros(
    _ shape: Shape2.Tuple,
    order: StorageOrder = .C
) -> Tensor2 { zeros(shape, DType.self, order) }

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape2.Tuple,
    dtype: Element.Type
) -> Tensor<Shape2, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype)
}

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape2.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape2, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype, order)
}

//---------------------------------------
// Rank3
@inlinable public func zeros(
    _ shape: Shape3.Tuple,
    order: StorageOrder = .C
) -> Tensor3 { zeros(shape, DType.self, order) }

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape3.Tuple,
    dtype: Element.Type
) -> Tensor<Shape3, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype)
}

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape3.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape3, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype, order)
}

//---------------------------------------
// Rank4
@inlinable public func zeros(
    _ shape: Shape4.Tuple,
    order: StorageOrder = .C
) -> Tensor4 { zeros(shape, DType.self, order) }

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape4.Tuple,
    dtype: Element.Type
) -> Tensor<Shape4, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype)
}

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape4.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape4, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype, order)
}

//---------------------------------------
// Rank5
@inlinable public func zeros(
    _ shape: Shape5.Tuple,
    order: StorageOrder = .C
) -> Tensor5 { zeros(shape, DType.self, order) }

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape5.Tuple,
    dtype: Element.Type
) -> Tensor<Shape5, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype)
}

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape5.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape5, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype, order)
}

//---------------------------------------
// Rank6
@inlinable public func zeros(
    _ shape: Shape6.Tuple,
    order: StorageOrder = .C
) -> Tensor6 { zeros(shape, DType.self, order) }

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape6.Tuple,
    dtype: Element.Type
) -> Tensor<Shape6, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype)
}

@inlinable public func zeros<Element: Numeric>(
    _ shape: Shape6.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor<Shape6, Element>
    where Element: StorageElement, Element.Value: Numeric
{
    zeros(shape, dtype, order)
}


//==============================================================================
/// zeros(like:
/// Return a new tensor of given shape and type filled with `value`
/// - Parameters:
///  - prototype: unspecified attributes are copied from this tensor
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
///  - shape: Int or tuple of Int
///    Shape of the zeros array, e.g., (2, 3) or 2.
/// - Returns: Fill of uninitialized (arbitrary) data of the given shape,
///   dtype, and order. Elements will not be initialized.

// same type and shape
@inlinable public func zeros<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil
) -> Tensor<S,E>
    where E: StorageElement, E.Value: Numeric
{
    zeros(prototype.shape, E.self, order ?? prototype.storage.order)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@inlinable public func zeros<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, E>
    where E: StorageElement, E.Value: Numeric
{
    assert(prototype.count == Shape1(shape).elementCount())
    return zeros(shape, E.self, order ?? prototype.storage.order)
}

// Rank2
@inlinable public func zeros<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, E>
    where E: StorageElement, E.Value: Numeric
{
    assert(prototype.count == Shape2(shape).elementCount())
    return zeros(shape, E.self, order ?? prototype.storage.order)
}

// Rank3
@inlinable public func zeros<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, E>
    where E: StorageElement, E.Value: Numeric
{
    assert(prototype.count == Shape3(shape).elementCount())
    return zeros(shape, E.self, order ?? prototype.storage.order)
}

// Rank4
@inlinable public func zeros<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, E>
    where E: StorageElement, E.Value: Numeric
{
    assert(prototype.count == Shape4(shape).elementCount())
    return zeros(shape, E.self, order ?? prototype.storage.order)
}

// Rank5
@inlinable public func zeros<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, E>
    where E: StorageElement, E.Value: Numeric
{
    assert(prototype.count == Shape5(shape).elementCount())
    return zeros(shape, E.self, order ?? prototype.storage.order)
}

// Rank6
@inlinable public func zeros<S,E>(
    like prototype: Tensor<S,E>,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, E>
    where E: StorageElement, E.Value: Numeric
{
    assert(prototype.count == Shape6(shape).elementCount())
    return zeros(shape, E.self, order ?? prototype.storage.order)
}


//------------------------------------------------------------------------------
// different type same shape
@inlinable public func zeros<S,E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<S, Element> {
    zeros(prototype.shape, Element.self, order ?? prototype.storage.order)
}

//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@inlinable public func zeros<S,E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element> {
    assert(prototype.count == Shape1(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storage.order)
}

// Rank2
@inlinable public func zeros<S,E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element> {
    assert(prototype.count == Shape2(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storage.order)
}

// Rank3
@inlinable public func zeros<S,E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element> {
    assert(prototype.count == Shape3(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storage.order)
}

// Rank4
@inlinable public func zeros<S,E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element> {
    assert(prototype.count == Shape4(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storage.order)
}

// Rank5
@inlinable public func zeros<S,E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element> {
    assert(prototype.count == Shape5(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storage.order)
}

// Rank6
@inlinable public func zeros<S,E, Element: Numeric>(
    like prototype: Tensor<S,E>,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element> {
    assert(prototype.count == Shape6(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storage.order)
}

