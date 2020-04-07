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

//==============================================================================
// ranked convenience types
/// Tensor
public typealias Tensor1<Element> = Tensor<Shape1, Element>
public typealias Tensor2<Element> = Tensor<Shape2, Element>
public typealias Tensor3<Element> = Tensor<Shape3, Element>
public typealias Tensor4<Element> = Tensor<Shape4, Element>
public typealias Tensor5<Element> = Tensor<Shape5, Element>
public typealias Tensor6<Element> = Tensor<Shape6, Element>

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
@inlinable
public func empty<Shape, Element>(
    _ shape: Shape.Tuple,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element> where Shape: TensorShape
{
    empty(Shape(shape), dtype, order)
}

@inlinable
public func empty<Shape, Element>(
    _ shape: Shape,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element> where Shape: TensorShape
{
    Tensor(shape, order: order)
}

//---------------------------------------
// Rank 0
@inlinable
public func empty() -> Tensor1<DType> {
    empty(Shape1(1), DType.self)
}

@inlinable
public func empty<Element>(dtype: Element.Type)
    -> Tensor1<Element> { empty(Shape1(1), dtype) }

//---------------------------------------
// Rank1
@inlinable
public func empty(_ shape: Shape1.Tuple, order: StorageOrder = .C)
    -> Tensor1<DType> { empty(shape, DType.self, order) }

@inlinable
public func empty<Element>(_ shape: Shape1.Tuple, dtype: Element.Type)
    -> Tensor1<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Shape1.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor1<Element> { empty(shape, dtype, order) }
    
//---------------------------------------
// Rank2
@inlinable
public func empty(_ shape: Shape2.Tuple, order: StorageOrder = .C)
    -> Tensor2<DType> { empty(shape, DType.self, order) }

@inlinable
public func empty<Element>(_ shape: Shape2.Tuple, dtype: Element.Type)
    -> Tensor2<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Shape2.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor2<Element> { empty(shape, dtype, order) }
    
//---------------------------------------
// Rank3
@inlinable
public func empty(_ shape: Shape3.Tuple, order: StorageOrder = .C)
    -> Tensor3<DType> { empty(shape, DType.self, order) }

@inlinable
public func empty<Element>(_ shape: Shape3.Tuple, dtype: Element.Type)
    -> Tensor3<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Shape3.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor3<Element> { empty(shape, dtype, order) }
    
//---------------------------------------
// Rank4
@inlinable
public func empty(_ shape: Shape4.Tuple, order: StorageOrder = .C)
    -> Tensor4<DType> { empty(shape, DType.self, order) }

@inlinable
public func empty<Element>(_ shape: Shape4.Tuple, dtype: Element.Type)
    -> Tensor4<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Shape4.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor4<Element> { empty(shape, dtype, order) }
    
//---------------------------------------
// Rank5
@inlinable
public func empty(_ shape: Shape5.Tuple, order: StorageOrder = .C)
    -> Tensor5<DType> { empty(shape, DType.self, order) }

@inlinable
public func empty<Element>(_ shape: Shape5.Tuple, dtype: Element.Type)
    -> Tensor5<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Shape5.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor5<Element> { empty(shape, dtype, order) }
    
//---------------------------------------
// Rank6
@inlinable
public func empty(_ shape: Shape6.Tuple, order: StorageOrder = .C)
    -> Tensor6<DType> { empty(shape, DType.self, order) }

@inlinable
public func empty<Element>(_ shape: Shape6.Tuple, dtype: Element.Type)
    -> Tensor6<Element> { empty(shape, dtype) }

@inlinable
public func empty<Element>(_ shape: Shape6.Tuple, dtype: Element.Type,
                           order: StorageOrder = .C)
    -> Tensor6<Element> { empty(shape, dtype, order) }
    

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
@inlinable
public func empty<T>(
    like prototype: T,
    order: StorageOrder? = nil
) -> Tensor<T.Shape, T.Element> where T: TensorType
{
    empty(prototype.shape, T.Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@inlinable public func empty<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return empty(shape, T.Element.self, order ?? prototype.storageOrder)
}
// Rank2
@inlinable public func empty<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return empty(shape, T.Element.self, order ?? prototype.storageOrder)
}
// Rank3
@inlinable public func empty<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return empty(shape, T.Element.self, order ?? prototype.storageOrder)
}
// Rank4
@inlinable public func empty<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return empty(shape, T.Element.self, order ?? prototype.storageOrder)
}
// Rank5
@inlinable public func empty<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return empty(shape, T.Element.self, order ?? prototype.storageOrder)
}
// Rank6
@inlinable public func empty<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return empty(shape, T.Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// different type same shape
@inlinable public func empty<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<T.Shape, Element> where T: TensorType
{
    empty(prototype.shape, Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@inlinable public func empty<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storageOrder)
}
// Rank2
@inlinable public func empty<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storageOrder)
}
// Rank3
@inlinable public func empty<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storageOrder)
}
// Rank4
@inlinable public func empty<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storageOrder)
}
// Rank5
@inlinable public func empty<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storageOrder)
}
// Rank6
@inlinable public func empty<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return empty(shape, Element.self, order ?? prototype.storageOrder)
}

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
    Tensor<Shape, Element>(repeating: value, shape, order: order)
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
@inlinable
public func full<Shape, Element>(
    _ shape: Shape.Tuple,
    _ value: Element,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element> where Shape: TensorShape
{
    full(Shape(shape), value, dtype, order)
}

@inlinable
public func full<Shape, Element>(
    _ shape: Shape,
    _ value: Element,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element> where Shape: TensorShape
{
    Tensor<Shape, Element>(shape, order: order).filled(with: value)
}

//---------------------------------------
// Rank0
@inlinable
public func full(_ value: DType) -> Tensor1<DType> {
    full(Shape1(1), value, DType.self)
}

@inlinable
public func full<Element>(_ value: Element, dtype: Element.Type)
    -> Tensor1<Element> { full(Shape1(1), value, dtype) }

//---------------------------------------
// Rank1
@inlinable
public func full(
    _ shape: Shape1.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor1<DType> { full(shape, value, DType.self, order) }

@inlinable
public func full<Element>(
    _ shape: Shape1.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor1<Element> { full(shape, value, dtype) }

@inlinable
public func full<Element>(
    _ shape: Shape1.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor1<Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank2
@inlinable
public func full(
    _ shape: Shape2.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor2<DType> { full(shape, value, DType.self, order) }

@inlinable
public func full<Element>(
    _ shape: Shape2.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor2<Element> { full(shape, value, dtype) }

@inlinable
public func full<Element>(
    _ shape: Shape2.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor2<Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank3
@inlinable
public func full(
    _ shape: Shape3.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor3<DType> { full(shape, value, DType.self, order) }

@inlinable
public func full<Element>(
    _ shape: Shape3.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor3<Element> { full(shape, value, dtype) }

@inlinable
public func full<Element>(
    _ shape: Shape3.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor3<Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank4
@inlinable
public func full(
    _ shape: Shape4.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor4<DType> { full(shape, value, DType.self, order) }

@inlinable
public func full<Element>(
    _ shape: Shape4.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor4<Element> { full(shape, value, dtype) }

@inlinable
public func full<Element>(
    _ shape: Shape4.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor4<Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank5
@inlinable
public func full(
    _ shape: Shape5.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor5<DType> { full(shape, value, DType.self, order) }

@inlinable
public func full<Element>(
    _ shape: Shape5.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor5<Element> { full(shape, value, dtype) }

@inlinable
public func full<Element>(
    _ shape: Shape5.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor5<Element> { full(shape, value, dtype, order) }

//---------------------------------------
// Rank6
@inlinable
public func full(
    _ shape: Shape6.Tuple,
    _ value: DType,
    order: StorageOrder = .C
) -> Tensor6<DType> { full(shape, value, DType.self, order) }

@inlinable
public func full<Element>(
    _ shape: Shape6.Tuple,
    _ value: Element,
    dtype: Element.Type
) -> Tensor6<Element> { full(shape, value, dtype) }

@inlinable
public func full<Element>(
    _ shape: Shape6.Tuple,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor6<Element> { full(shape, value, dtype, order) }


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
@inlinable
public func full<T>(
    like prototype: T,
    _ value: T.Element,
    order: StorageOrder? = nil
) -> Tensor<T.Shape, T.Element> where T: TensorType
{
    full(prototype.shape, value, T.Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@inlinable public func full<T>(
    like prototype: T,
    _ value: T.Element,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return full(shape, value, T.Element.self, order ?? prototype.storageOrder)
}

// Rank2
@inlinable public func full<T>(
    like prototype: T,
    _ value: T.Element,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return full(shape, value, T.Element.self, order ?? prototype.storageOrder)
}

// Rank3
@inlinable public func full<T>(
    like prototype: T,
    _ value: T.Element,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return full(shape, value, T.Element.self, order ?? prototype.storageOrder)
}

// Rank4
@inlinable public func full<T>(
    like prototype: T,
    _ value: T.Element,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return full(shape, value, T.Element.self, order ?? prototype.storageOrder)
}

// Rank5
@inlinable public func full<T>(
    like prototype: T,
    _ value: T.Element,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return full(shape, value, T.Element.self, order ?? prototype.storageOrder)
}

// Rank6
@inlinable public func full<T>(
    like prototype: T,
    _ value: T.Element,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, T.Element> where T: TensorType
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return full(shape, value, T.Element.self, order ?? prototype.storageOrder)
}


//------------------------------------------------------------------------------
// different type same shape
@inlinable
public func full<T, Element>(
    like prototype: T,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<T.Shape, Element> where T: TensorType
{
    full(prototype.shape, value, Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@inlinable public func full<T, Element>(
    like prototype: T,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storageOrder)
}

// Rank2
@inlinable public func full<T, Element>(
    like prototype: T,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storageOrder)
}

// Rank3
@inlinable public func full<T, Element>(
    like prototype: T,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storageOrder)
}

// Rank4
@inlinable public func full<T, Element>(
    like prototype: T,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storageOrder)
}

// Rank5
@inlinable public func full<T, Element>(
    like prototype: T,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storageOrder)
}

// Rank6
@inlinable public func full<T, Element>(
    like prototype: T,
    _ value: Element,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element> where T: TensorType
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return full(shape, value, Element.self, order ?? prototype.storageOrder)
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
@inlinable
public func ones<Shape, Element>(
    _ shape: Shape.Tuple,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element: Numeric
{
    ones(Shape(shape), dtype, order)
}

@inlinable
public func ones<Shape, Element>(
    _ shape: Shape,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element: Numeric
{
    Tensor<Shape, Element>(1, shape, order: order)
}

//---------------------------------------
// Rank0
@inlinable
public func ones() -> Tensor1<DType> {
    ones(Shape1(1), DType.self)
}

@inlinable
public func ones<Element>(dtype: Element.Type)
    -> Tensor1<Element> where Element: Numeric { ones(Shape1(1), dtype) }

//---------------------------------------
// Rank1
@inlinable
public func ones(_ shape: Shape1.Tuple, order: StorageOrder = .C)
    -> Tensor1<DType> { ones(shape, DType.self, order) }

@inlinable
public func ones<Element>(_ shape: Shape1.Tuple, dtype: Element.Type)
    -> Tensor1<Element> where Element: Numeric { ones(shape, dtype) }

@inlinable
public func ones<Element>(
    _ shape: Shape1.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor1<Element> where Element: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank2
@inlinable
public func ones(_ shape: Shape2.Tuple, order: StorageOrder = .C)
    -> Tensor2<DType> { ones(shape, DType.self, order) }

@inlinable
public func ones<Element>(_ shape: Shape2.Tuple, dtype: Element.Type)
    -> Tensor2<Element> where Element: Numeric { ones(shape, dtype) }

@inlinable
public func ones<Element>(
    _ shape: Shape2.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor2<Element> where Element: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank3
@inlinable
public func ones(_ shape: Shape3.Tuple, order: StorageOrder = .C)
    -> Tensor3<DType> { ones(shape, DType.self, order) }

@inlinable
public func ones<Element>(_ shape: Shape3.Tuple, dtype: Element.Type)
    -> Tensor3<Element> where Element: Numeric { ones(shape, dtype) }

@inlinable
public func ones<Element>(
    _ shape: Shape3.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor3<Element> where Element: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank4
@inlinable
public func ones(_ shape: Shape4.Tuple, order: StorageOrder = .C)
    -> Tensor4<DType> { ones(shape, DType.self, order) }

@inlinable
public func ones<Element>(_ shape: Shape4.Tuple, dtype: Element.Type)
    -> Tensor4<Element> where Element: Numeric { ones(shape, dtype) }

@inlinable
public func ones<Element>(
    _ shape: Shape4.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor4<Element> where Element: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank5
@inlinable
public func ones(_ shape: Shape5.Tuple, order: StorageOrder = .C)
    -> Tensor5<DType> { ones(shape, DType.self, order) }

@inlinable
public func ones<Element>(_ shape: Shape5.Tuple, dtype: Element.Type)
    -> Tensor5<Element> where Element: Numeric { ones(shape, dtype) }

@inlinable
public func ones<Element>(
    _ shape: Shape5.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor5<Element> where Element: Numeric { ones(shape, dtype, order) }

//---------------------------------------
// Rank6
@inlinable
public func ones(_ shape: Shape6.Tuple, order: StorageOrder = .C)
    -> Tensor6<DType> { ones(shape, DType.self, order) }

@inlinable
public func ones<Element>(_ shape: Shape6.Tuple, dtype: Element.Type)
    -> Tensor6<Element> where Element: Numeric { ones(shape, dtype) }

@inlinable
public func ones<Element>(
    _ shape: Shape6.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor6<Element> where Element: Numeric { ones(shape, dtype, order) }


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
@inlinable
public func ones<T>(like prototype: T, order: StorageOrder? = nil)
    -> Tensor<T.Shape, T.Element>
    where T: TensorType, T.Element: Numeric
{
    ones(prototype.shape, T.Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@inlinable public func ones<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return ones(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank2
@inlinable public func ones<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return ones(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank3
@inlinable public func ones<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return ones(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank4
@inlinable public func ones<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return ones(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank5
@inlinable public func ones<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return ones(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank6
@inlinable public func ones<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return ones(shape, T.Element.self, order ?? prototype.storageOrder)
}


//------------------------------------------------------------------------------
// different type same shape
@inlinable
public func ones<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<T.Shape, Element>
    where T: TensorType, Element: Numeric
{
    ones(prototype.shape, Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@inlinable public func ones<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank2
@inlinable public func ones<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank3
@inlinable public func ones<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank4
@inlinable public func ones<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank5
@inlinable public func ones<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank6
@inlinable public func ones<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return ones(shape, Element.self, order ?? prototype.storageOrder)
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
@inlinable
public func zeros<Shape, Element>(
    _ shape: Shape.Tuple,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element: Numeric
{
    zeros(Shape(shape), dtype, order)
}

@inlinable
public func zeros<Shape, Element>(
    _ shape: Shape,
    _ dtype: Element.Type,
    _ order: StorageOrder = .C
) -> Tensor<Shape, Element>
    where Shape: TensorShape, Element: Numeric
{
    Tensor<Shape, Element>(0, shape, order: order)
}

//---------------------------------------
// Rank0
@inlinable
public func zeros() -> Tensor1<DType> {
    zeros(Shape1(1), DType.self)
}

@inlinable
public func zeros<Element>(dtype: Element.Type)
    -> Tensor1<Element> where Element: Numeric { zeros(Shape1(1), dtype) }

//---------------------------------------
// Rank1
@inlinable
public func zeros(_ shape: Shape1.Tuple, order: StorageOrder = .C)
    -> Tensor1<DType> { zeros(shape, DType.self, order) }

@inlinable
public func zeros<Element>(_ shape: Shape1.Tuple, dtype: Element.Type)
    -> Tensor1<Element> where Element: Numeric { zeros(shape, dtype) }

@inlinable
public func zeros<Element>(
    _ shape: Shape1.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor1<Element> where Element: Numeric { zeros(shape, dtype, order) }

//---------------------------------------
// Rank2
@inlinable
public func zeros(_ shape: Shape2.Tuple, order: StorageOrder = .C)
    -> Tensor2<DType> { zeros(shape, DType.self, order) }

@inlinable
public func zeros<Element>(_ shape: Shape2.Tuple, dtype: Element.Type)
    -> Tensor2<Element> where Element: Numeric { zeros(shape, dtype) }

@inlinable
public func zeros<Element>(
    _ shape: Shape2.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor2<Element> where Element: Numeric { zeros(shape, dtype, order) }

//---------------------------------------
// Rank3
@inlinable
public func zeros(_ shape: Shape3.Tuple, order: StorageOrder = .C)
    -> Tensor3<DType> { zeros(shape, DType.self, order) }

@inlinable
public func zeros<Element>(_ shape: Shape3.Tuple, dtype: Element.Type)
    -> Tensor3<Element> where Element: Numeric { zeros(shape, dtype) }

@inlinable
public func zeros<Element>(
    _ shape: Shape3.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor3<Element> where Element: Numeric { zeros(shape, dtype, order) }

//---------------------------------------
// Rank4
@inlinable
public func zeros(_ shape: Shape4.Tuple, order: StorageOrder = .C)
    -> Tensor4<DType> { zeros(shape, DType.self, order) }

@inlinable
public func zeros<Element>(_ shape: Shape4.Tuple, dtype: Element.Type)
    -> Tensor4<Element> where Element: Numeric { zeros(shape, dtype) }

@inlinable
public func zeros<Element>(
    _ shape: Shape4.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor4<Element> where Element: Numeric { zeros(shape, dtype, order) }

//---------------------------------------
// Rank5
@inlinable
public func zeros(_ shape: Shape5.Tuple, order: StorageOrder = .C)
    -> Tensor5<DType> { zeros(shape, DType.self, order) }

@inlinable
public func zeros<Element>(_ shape: Shape5.Tuple, dtype: Element.Type)
    -> Tensor5<Element> where Element: Numeric { zeros(shape, dtype) }

@inlinable
public func zeros<Element>(
    _ shape: Shape5.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor5<Element> where Element: Numeric { zeros(shape, dtype, order) }

//---------------------------------------
// Rank6
@inlinable
public func zeros(_ shape: Shape6.Tuple, order: StorageOrder = .C)
    -> Tensor6<DType> { zeros(shape, DType.self, order) }

@inlinable
public func zeros<Element>(_ shape: Shape6.Tuple, dtype: Element.Type)
    -> Tensor6<Element> where Element: Numeric { zeros(shape, dtype) }

@inlinable
public func zeros<Element>(
    _ shape: Shape6.Tuple,
    dtype: Element.Type,
    order: StorageOrder = .C
) -> Tensor6<Element> where Element: Numeric { zeros(shape, dtype, order) }


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
@inlinable
public func zeros<T>(like prototype: T, order: StorageOrder? = nil)
    -> Tensor<T.Shape, T.Element>
    where T: TensorType, T.Element: Numeric
{
    zeros(prototype.shape, T.Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// same type different shape
// Rank1
@inlinable public func zeros<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return zeros(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank2
@inlinable public func zeros<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return zeros(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank3
@inlinable public func zeros<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return zeros(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank4
@inlinable public func zeros<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return zeros(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank5
@inlinable public func zeros<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return zeros(shape, T.Element.self, order ?? prototype.storageOrder)
}

// Rank6
@inlinable public func zeros<T>(
    like prototype: T,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, T.Element>
    where T: TensorType, T.Element: Numeric
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return zeros(shape, T.Element.self, order ?? prototype.storageOrder)
}


//------------------------------------------------------------------------------
// different type same shape
@inlinable
public func zeros<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil
) -> Tensor<T.Shape, Element>
    where T: TensorType, Element: Numeric
{
    zeros(prototype.shape, Element.self, order ?? prototype.storageOrder)
}

//------------------------------------------------------------------------------
// different type, different shape
// Rank1
@inlinable public func zeros<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape1.Tuple
) -> Tensor<Shape1, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape1(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank2
@inlinable public func zeros<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape2.Tuple
) -> Tensor<Shape2, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape2(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank3
@inlinable public func zeros<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape3.Tuple
) -> Tensor<Shape3, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape3(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank4
@inlinable public func zeros<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape4.Tuple
) -> Tensor<Shape4, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape4(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank5
@inlinable public func zeros<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape5.Tuple
) -> Tensor<Shape5, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape5(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storageOrder)
}

// Rank6
@inlinable public func zeros<T, Element>(
    like prototype: T,
    dtype: Element.Type,
    order: StorageOrder? = nil,
    shape: Shape6.Tuple
) -> Tensor<Shape6, Element>
    where T: TensorType, Element: Numeric
{
    assert(prototype.elementCount == Shape6(shape).elementCount())
    return zeros(shape, Element.self, order ?? prototype.storageOrder)
}

