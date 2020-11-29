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
import Numerics

//==============================================================================
// ranked convenience types
/// Tensor
public typealias TensorR1<Element: StorageElement> = Tensor<Shape1, Element>
public typealias TensorR2<Element: StorageElement> = Tensor<Shape2, Element>
public typealias TensorR3<Element: StorageElement> = Tensor<Shape3, Element>
public typealias TensorR4<Element: StorageElement> = Tensor<Shape4, Element>
public typealias TensorR5<Element: StorageElement> = Tensor<Shape5, Element>
public typealias TensorR6<Element: StorageElement> = Tensor<Shape6, Element>

//==============================================================================
// parameter matching helpers
@usableFromInline func repeatedStrides<Shape, E>(
  matching other: Tensor<Shape, E>,
  to shape: Shape
) -> Shape {
  // make sure the bounds are compatible
  assert(
    {
      for i in 0..<Shape.rank {
        if other.shape[i] != 1 && shape[i] != other.shape[i] {
          return false
        }
      }
      return true
    }(), "repeated tensor dimensions must be either 1" + " or match the repeated shape")

  // compute strides, setting stride to 0 for repeated dimensions
  var repeatedStrides = Shape.zero
  for i in 0..<Shape.rank where other.shape[i] == shape[i] {
    repeatedStrides[i] = other.strides[i]
  }
  return repeatedStrides
}

//==============================================================================
// Tensor initializers
//==============================================================================
extension Tensor {
  //--------------------------------------------------------------------------
  /// init(shape:order:
  /// creates a dense shape
  /// - Parameters:
  ///  - shape: the n-dimensional shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init(
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) {
    let count = shape.elementCount()
    let storage = Platform.Storage(type: TensorElement.self, count: count, name: name)
    self.init(
      shape: shape,
      strides: shape.strides(for: order),
      count: count,
      storage: storage,
      storageBase: 0,
      spanCount: count,
      order: order,
      shared: false)
  }

  //--------------------------------------------------------------------------
  /// init(like:
  /// convenience initializer to initialize with the shape and type as `other`
  /// - Parameters:
  ///  - other: a tensor to copy attributes from
  @inlinable public init(like other: Self) {
    self.init(shape: other.shape, order: other.order)
  }

  //--------------------------------------------------------------------------
  /// init(element:
  /// creates a tensor with a single element value
  /// - Parameters:
  ///  - element: the element value for the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init(
    _ element: TensorElement.Value,
    order: Order = .defaultOrder,
    name: String = defaultElementName
  ) {
    self.init(single: element, shape: Shape.one, order: order, name: name)
  }

  //--------------------------------------------------------------------------
  /// init(repeating element:shape:
  /// Repeats a single stored element while indexing
  /// - Parameters:
  ///  - element: the element value to repeat while indexing
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init(
    repeating element: TensorElement.Value,
    to shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultElementName
  ) {
    self.init(single: element, shape: shape, order: order, name: name)
  }

  //--------------------------------------------------------------------------
  /// init(repeating other:shape:
  /// Repeats a tensor withing the specified shape while indexing
  /// - Parameters:
  ///  - other: the tensor to repeat
  ///  - shape: the shape of the tensor
  @inlinable public init(repeating other: Self, to shape: Shape) {
    let strides = repeatedStrides(matching: other, to: shape)
    let count = shape.elementCount()
    self.init(
      shape: shape,
      strides: strides,
      count: count,
      storage: other.storage,
      storageBase: other.storageBase,
      spanCount: shape.spanCount(stridedBy: strides),
      order: other.order,
      shared: other.isShared)
  }

  //--------------------------------------------------------------------------
  /// reductionShape
  /// computes the shape for a reduction result along the specified axes
  @inlinable public func reductionShape(along axes: [Int]?) -> Shape {
    guard let axes = axes else { return Shape.one }
    assert(
      Set(axes).isSubset(of: (-Shape.rank)..<Shape.rank),
      "axis must be within -rank..<rank")
    var result = shape
    // set 1 for each dimension specified in the axes list
    axes.forEach {
      result[$0 >= 0 ? $0 : $0 + Shape.rank] = 1
    }
    return result
  }
}

//==============================================================================
// Tensor range initializers
//==============================================================================

extension Tensor {
  //--------------------------------------------------------------------------
  /// `init(range:shape:order:name:`
  /// initializes the tensor with a progressive logical index value
  ///
  /// - Parameters:
  ///  - range: the index range. The number of elements in the range
  ///    must be equal to the number of elements described by shape.
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init(
    range: Range<Int>,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) where TensorElement.Value: Numeric {
    self.init(shape: shape, order: order, name: name)
    fill(&self, with: range)
  }

  //--------------------------------------------------------------------------
  /// `init(from:to:shape:order:name:`
  /// initializes the tensor with a progressive logical index value
  ///
  /// - Parameters:
  ///  - from: the initial value in the range
  ///  - to: the last value in the range
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init(
    from first: TensorElement.Value,
    to last: TensorElement.Value,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) where TensorElement.Value: BinaryInteger {
    self.init(shape: shape, order: order, name: name)
    let step = (last - first) / TensorElement.Value(exactly: count - 1)!
    currentQueue.fill(&self, from: first, to: last, by: step)
  }

  @inlinable public init(
    from first: TensorElement.Value,
    to last: TensorElement.Value,
    shape: Shape.Tuple,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) where TensorElement.Value: BinaryInteger {
    self.init(from: first, to: last, shape: Shape(shape), order: order, name: name)
  }
  
  @inlinable public init(
    from first: TensorElement.Value,
    to last: TensorElement.Value,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) where TensorElement.Value: AlgebraicField {
    self.init(shape: shape, order: order, name: name)
    let step = (last - first) / TensorElement.Value(exactly: count - 1)!
    currentQueue.fill(&self, from: first, to: last, by: step)
  }

  @inlinable public init(
    from first: TensorElement.Value,
    to last: TensorElement.Value,
    shape: Shape.Tuple,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) where TensorElement.Value: AlgebraicField {
    self.init(from: first, to: last, shape: Shape(shape), order: order, name: name)
  }
}

//==============================================================================
// Tensor collection initializers
//==============================================================================

extension Tensor {
  //--------------------------------------------------------------------------
  /// `init(stored:shape:order:name:`
  /// initializes the tensor storage buffer with `storage elements`
  ///
  /// - Parameters:
  ///  - elements: the collection used to initialize storage, where the
  ///  collection elements are equal to the `TensorElement.Stored` type.
  ///  For example:
  ///  For `Int4`, `TensorElement.Stored == UInt8`, so `C.Element`
  ///  is expected to be of type `UInt8`
  ///  For `Float`, `TensorElement.Stored == Float`, so `C.Element`
  ///  is expected to be of type `Float`
  ///  For `Float16`, `TensorElement.Stored == Float16`, so `C.Element`
  ///  is expected to be of type `Float16`
  ///
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init<C>(
    stored elements: C,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) where C: Collection, C.Element == TensorElement.Stored {
    self.init(shape: shape, order: order, name: name)
    let buffer = readWrite(using: Platform.syncQueue)
    assert(buffer.count == elements.count)
    _ = buffer.initialize(from: elements)
  }

  //--------------------------------------------------------------------------
  /// `init(elements:shape:order:name:`
  /// initializes the tensor with the given `elements while
  /// converting them to `TensorElement.Stored` type.
  /// For example in the `Float16` case, `C.Element` would be type `Float`
  /// but the stored type will be the `Float16` bit pattern.
  /// So floats go in, and a buffer with `Float16` pattern is
  /// ready for use on a discrete device with native `Float16` support.
  ///
  /// - Parameters:
  ///  - elements: the value collection used to initialize storage
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init<C>(
    _ elements: C,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) where C: Collection, C.Element == TensorElement.Value {
    assert(shape.elementCount() == elements.count)
    self.init(shape: shape, order: order, name: name)

    // get the storage buffer and set the values
    let buffer = readWrite(using: Platform.syncQueue)
    for (i, v) in elements.enumerated() {
      TensorElement.set(value: v, in: buffer, at: i)
    }
  }

  //--------------------------------------------------------------------------
  /// `init(elements:shape:order:name:`
  /// implicitly casts from C.Element Bool -> Numeric TensorElement.Value
  ///
  /// - Parameters:
  ///  - elements: the value collection used to initialize storage
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init<C>(
    _ elements: C,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) where C: Collection, C.Element == Bool, TensorElement.Value: Numeric {
    assert(shape.elementCount() == elements.count)
    self.init(shape: shape, order: order, name: name)

    // get the storage buffer and set the values
    let buffer = readWrite(using: Platform.syncQueue)
    for (i, v) in elements.enumerated() {
      TensorElement.set(
        value: Element(exactly: v ? 1 : 0)!,
        in: buffer, at: i)
    }
  }

  //--------------------------------------------------------------------------
  /// `init(elements:shape:order:name:`
  /// implicitly casts from C.Element Numeric -> Bool TensorElement.Value
  ///
  /// - Parameters:
  ///  - elements: the value collection used to initialize storage
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init<C>(
    _ elements: C,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) where C: Collection, C.Element: Numeric, TensorElement.Value == Bool {
    assert(shape.elementCount() == elements.count)
    self.init(shape: shape, order: order, name: name)

    // get the storage buffer and set the values
    let buffer = readWrite(using: Platform.syncQueue)
    for (i, v) in elements.enumerated() {
      TensorElement.set(value: Element(v != 0), in: buffer, at: i)
    }
  }

  //--------------------------------------------------------------------------
  /// `init(elements:shape:order:name:`
  /// implicitly casts from C.Element integer -> TensorElement.Value
  ///
  /// - Parameters:
  ///  - elements: the value collection used to initialize storage
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init<C>(
    _ elements: C,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  )
  where
    C: Collection, C.Element: BinaryInteger,
    TensorElement.Value: Numeric
  {
    assert(shape.elementCount() == elements.count)
    self.init(shape: shape, order: order, name: name)

    // get the storage buffer on the cpu and set the values
    let buffer = readWrite(using: Platform.syncQueue)
    for (i, v) in elements.enumerated() {
      TensorElement.set(value: Element(exactly: v)!, in: buffer, at: i)
    }
  }

  //--------------------------------------------------------------------------
  /// `init(elements:shape:order:name:`
  /// implicitly casts from floating C.Element -> floating TensorElement.Value
  ///
  /// - Parameters:
  ///  - elements: the value collection used to initialize storage
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  // Note: to handle the case of Double <--> Float
  @inlinable public init<C>(
    _ elements: C,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  )
  where
    C: Collection, C.Element: BinaryFloatingPoint,
    TensorElement.Value: BinaryFloatingPoint
  {
    assert(shape.elementCount() == elements.count)
    self.init(shape: shape, order: order, name: name)

    // get the storage buffer and set the values
    let buffer = readWrite(using: Platform.syncQueue)
    for (i, v) in elements.enumerated() {
      TensorElement.set(value: Element(v), in: buffer, at: i)
    }
  }

  //--------------------------------------------------------------------------
  /// `init(elements:shape:order:name:`
  /// implicitly casts from C.Element float -> integer TensorElement.Value
  ///
  /// - Parameters:
  ///  - elements: the value collection used to initialize storage
  ///  - shape: the shape of the tensor
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  // Note: to handle the case of Double <--> Float
  @inlinable public init<C>(
    _ elements: C,
    shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  )
  where
    C: Collection, C.Element: BinaryFloatingPoint,
    TensorElement.Value: BinaryInteger
  {
    assert(shape.elementCount() == elements.count)
    self.init(shape: shape, order: order, name: name)

    // get the storage buffer and set the values
    let buffer = readWrite(using: Platform.syncQueue)
    for (i, v) in elements.enumerated() {
      TensorElement.set(value: Element(v), in: buffer, at: i)
    }
  }
}

//==============================================================================
// Rank transformations
extension Tensor {
  /// concatenated tensors
  @inlinable public init(concatenating tensors: Self..., axis: Int = 0) {
    self = Self(concatenating: tensors, axis: axis)
  }

  @inlinable public init(concatenating tensors: [Self], axis: Int = 0) {
    self = SwiftRTCore.concatenate(tensors, axis: axis)
  }
}

//==============================================================================
/// init(reshaping:shape:order:
/// - Parameters:
///  - other: the tensor to reshape
///  - newShape: the shape of the new tensor
///  - order: the storage order of the new tensor
extension Tensor {

  @inlinable public init<S>(
    reshaping other: Tensor<S, TensorElement>,
    to newShape: Shape,
    order newLayout: Order = .defaultOrder
  ) {
    assert(other.isContiguous, "cannot reshape non contiguous data")
    assert(
      {
        var found = false
        for i in 0..<Shape.rank where newShape[i] == -1 {
          if found { return false }
          found = true
        }
        return true
      }(), "There can only be one instance of -1 in the shape")

    // resolve an implied dimension if it exists
    var shape = newShape
    for i in 0..<Shape.rank where newShape[i] == -1 {
      shape[i] = 1
      let specifiedCount = shape.elementCount()
      assert(
        other.count % specifiedCount == 0,
        "incompatible dimensions")
      shape[i] = other.count / specifiedCount
    }
    assert(
      shape.elementCount() == other.count,
      "the new shape must have the same number of elements as other")

    // determine storage order
    let order: Order =
      newLayout == .col || (newLayout.rawValue == Order.A && other.order == .col) ? .col : .row

    // reorder other's elements if needed
    var source = other
    if order != other.order {
      // change the source to have the new storage order and copy
      let strides = other.shape.strides(for: order)
      source = Tensor<S, TensorElement>(
        shape: other.shape,
        strides: strides,
        count: other.count,
        storage: Platform.Storage(
          type: TensorElement.self,
          count: source.count,
          name: other.name),
        storageBase: 0,
        spanCount: other.spanCount,
        order: order,
        shared: other.isShared)

      // performs an indexed copy which reorders the elements
      currentQueue.diagnostic(
        .reorder,
        "copying \(other.name) order: \(other.order) --> "
          + "\(source.name) \(Element.self)[\(source.count)] "
          + "order: \(source.order) on \(currentQueue.name)",
        categories: [.dataCopy, .dataReorder])

      copyElements(from: other, to: &source)
    }

    // init with new shape in the corrected order
    self.init(
      shape: shape,
      strides: shape.strides(for: order),
      count: source.count,
      storage: source.storage,
      storageBase: source.storageBase,
      spanCount: source.spanCount,
      order: source.order,
      shared: source.isShared)
  }
}

//==============================================================================
/// init(expanding other:axes:
/// - Parameters:
///  - other: the tensor to expand
///  - axes: the list of axes to expand

extension Tensor {

  @inlinable public init<S, Axes>(
    expanding other: Tensor<S, TensorElement>,
    axes: Axes
  ) where Axes: TensorShape {
    // set the expanded axes
    var shape = Shape.zero
    var strides = Shape.zero

    // default case indents by 1
    if Axes.rank == 1 && axes[0] == 0 {
      shape[0] = 1
      strides[0] = other.shape[0] * other.strides[0]
      for (i, j) in zip(1..<Shape.rank, 0..<S.rank) {
        shape[i] = other.shape[j]
        strides[i] = other.strides[j]
      }
    } else {
      assert(Shape.rank == S.rank + Axes.rank, "rank mismatch")
      // set 1 in expanded dimensions making sure axes are positive
      // and keeping in mind the axes could be in any order
      for i in 0..<Axes.rank {
        shape[axes[i] >= 0 ? axes[i] : axes[i] + S.rank] = 1
      }

      var axis = Shape.rank - 1
      var otherAxis = S.rank - 1
      while axis >= 0 {
        if shape[axis] == 1 {
          if axis == Shape.rank - 1 {
            // if the last dimension is expanded, then stride is 1
            strides[axis] = 1
          } else {
            // if inserted, then compute stride
            strides[axis] = shape[axis + 1] * strides[axis + 1]
          }
        } else {
          // simply copy stride
          shape[axis] = other.shape[otherAxis]
          strides[axis] = other.strides[otherAxis]
          otherAxis -= 1
        }
        axis -= 1
      }
    }

    self.init(
      shape: shape,
      strides: strides,
      count: other.count,
      storage: other.storage,
      storageBase: other.storageBase,
      spanCount: other.spanCount,
      order: other.order,
      shared: other.isShared)
  }

  @inlinable public init<S>(
    expanding other: Tensor<S, TensorElement>,
    axes: Int...
  ) {
    self.init(expanding: other, axes: Shape(axes))
  }
}

//==============================================================================
/// init(squeezing:axes
/// - Parameters:
///  - other: the collection to squeeze
///  - axes: a list of axes to squeeze
extension Tensor {

  @inlinable public init<S, Axes>(
    squeezing other: Tensor<S, TensorElement>,
    axes: Axes
  ) where Axes: TensorShape {
    assert(Shape.rank == S.rank - Axes.rank, "rank mismatch")
    var axis = 0
    var shape = Shape.zero
    var strides = Shape.zero
    var otherShape = other.shape

    // zero the axes to squeeze. These are done in two loops
    // because the axes list could be in any order
    for i in 0..<Axes.rank {
      otherShape[axes[i] >= 0 ? axes[i] : axes[i] + S.rank] = 0
    }

    for i in 0..<S.rank where otherShape[i] > 0 {
      shape[axis] = otherShape[i]
      strides[axis] = other.strides[i]
      axis += 1
    }

    self.init(
      shape: shape,
      strides: strides,
      count: other.count,
      storage: other.storage,
      storageBase: other.storageBase,
      spanCount: other.spanCount,
      order: other.order,
      shared: other.isShared)
  }

  @inlinable public init<S>(
    squeezing other: Tensor<S, TensorElement>,
    axes: Int...
  ) {
    self.init(squeezing: other, axes: Shape(axes))
  }
}

//==============================================================================
/// init(stacking:axis:
/// - Parameters:
///  - others: the collection to squeeze
///  - axis: the axis to stack along
extension Tensor {

  @inlinable public init<S>(
    stacking others: [Tensor<S, TensorElement>],
    axis: Int = 0
  ) {
    // make positive
    let positiveAxis = axis < 0 ? axis + Shape.rank : axis
    // create tensor of stacked shape and copy
    self = Self(
        shape: stackedShape(of: others, along: positiveAxis),
        order: others[0].order)
    stack(others, axis: positiveAxis, into: &self)
  }

  @inlinable public init<S>(
    stacking others: Tensor<S, TensorElement>...,
    axis: Int = 0
  ) {
    self.init(stacking: others, axis: axis)
  }
}

//==============================================================================
/// stackedShape(
// get the stacked shape with the inserted axis
@inlinable func stackedShape<S, SR, E>(
  of tensors: [Tensor<S, E>],
  along axis: Int = 0
) -> SR where SR: TensorShape {
  assert(axis >= 0)
  var j = 0
  var stackedShape = SR.zero
  stackedShape[axis] = tensors.count
  for i in 0..<SR.rank where i != axis {
    stackedShape[i] = tensors[0].shape[j]
    j += 1
  }
  return stackedShape
}

//==============================================================================
/// stack(_:axis:into:
/// - Parameters:
///  - others: the collection to squeeze
///  - axis: the axis to stack along
///  - result: the output tensor
@inlinable public func stack<S, SR, E>(
  _ tensors: [Tensor<S, E>],
  axis: Int = 0,
  into result: inout Tensor<SR, E>
) {
  // make positive
  let axis = axis < 0 ? axis + SR.rank : axis

  // verify that tensors are the correct rank and same shape
  assert(
    tensors.count > 0 && S.rank == SR.rank - 1,
    "stacked tensors must be one less than result rank \(S.rank - 1)")
  assert(
    {
      let shape = tensors[0].shape
      for i in 1..<tensors.count {
        if tensors[i].shape != shape { return false }
      }
      return true
    }(), "stacked tensors must all be the same size")
  assert(
    result.shape == stackedShape(of: tensors, along: axis),
    "result tensor does not match the stacked shape of the inputs")

  // expand dimensions of each input along axis
  let expanded = tensors.map {
    Tensor<SR, E>(expanding: $0, axes: Shape1(axis))
  }

  // copy others into place
  var lower = SR.zero
  for tensor in expanded {
    result[lower, lower &+ tensor.shape] = tensor
    lower[axis] += 1
  }
}

//==============================================================================
/// init(indenting:
///
extension Tensor {

  @inlinable public init<S>(
    indenting other: Tensor<S, TensorElement>
  ) {
    assert(S.rank < Shape.rank, "can only indent lower ranked shapes")

    // Self and other are different ranks so we append other's elements
    let start = Shape.rank - S.rank
    var shape = Shape.one
    var strides = Shape.one
    for (i, j) in zip(start..<Shape.rank, 0..<S.rank) {
      shape[i] = other.shape[j]
      strides[i] = other.strides[j]
    }
    for i in 0..<start {
      strides[i] = other.strides[0]
    }

    //-----------------------------------
    self.init(
      shape: shape,
      strides: strides,
      count: other.count,
      storage: other.storage,
      storageBase: other.storageBase,
      spanCount: other.spanCount,
      order: other.order,
      shared: other.isShared)
  }
}

//==============================================================================
/// init(padding:
///
extension Tensor {

  @inlinable public init<S>(
    padding other: Tensor<S, TensorElement>
  ) {
    assert(S.rank < Shape.rank, "can only indent lower ranked shapes")

    // Self and other are different ranks so we append other's elements
    var shape = Shape.one
    var strides = Shape.one
    for i in 0..<S.rank {
      shape[i] = other.shape[i]
      strides[i] = other.strides[i]
    }

    //-----------------------------------
    self.init(
      shape: shape,
      strides: strides,
      count: other.count,
      storage: other.storage,
      storageBase: other.storageBase,
      spanCount: other.spanCount,
      order: other.order,
      shared: other.isShared)
  }
}

//==============================================================================
/// init(transposing:permutations:
/// Returns a new data shape where the bounds and strides are permuted
/// - Parameter permutations: the indice order mapping. `count` must
///   equal `rank`
/// - Returns: transposed/permuted shape
/// - Precondition: Each value in `permutations` must be in the range
///   `-rank..<rank`
extension Tensor {

  @inlinable public init(
    transposing other: Self,
    permutatedBy permutations: Shape? = nil
  ) {
    assert(Shape.rank > 1, "can only transpose shapes greater than rank 1")

    func makePositive(dims: Shape) -> Shape {
      var positive = dims
      for i in 0..<Shape.rank where positive[i] < 0 {
        positive[i] += Shape.rank
      }
      return positive
    }

    // determine the new bounds and strides
    var shape = other.shape
    var strides = other.strides
    if let perm = permutations {
      let mapping = makePositive(dims: perm)
      for index in 0..<Shape.rank {
        shape[index] = other.shape[mapping[index]]
        strides[index] = other.strides[mapping[index]]
      }
    } else {
      // simple swap of last two dimensions
      shape.swapAt(Shape.rank - 1, Shape.rank - 2)
      strides.swapAt(Shape.rank - 1, Shape.rank - 2)
    }

    //-----------------------------------
    self.init(
      shape: shape,
      strides: strides,
      count: other.count,
      storage: other.storage,
      storageBase: other.storageBase,
      spanCount: other.spanCount,
      order: other.order,
      shared: other.isShared)
  }

  /// - Returns: transpose of self
  @inlinable public var t: Self { Self(transposing: self) }

  @inlinable public func transposed(permutatedBy permutations: Shape) -> Self {
    Self(transposing: self, permutatedBy: permutations)
  }
}

//==============================================================================
//
extension Tensor where TensorElement.Value: Numeric {
  //--------------------------------------------------------------------------
  /// init(zeros shape:order:
  /// creates a dense shape filled with zeros
  /// - Parameters:
  ///  - shape: the n-dimensional shape of the tensor to be filled
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init(
    zeros shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) {
    self.init(shape: shape, order: order, name: name)
    fill(&self, with: 0)
  }

  //--------------------------------------------------------------------------
  /// init(ones shape:order:
  /// creates a dense shape filled with ones
  /// - Parameters:
  ///  - shape: the n-dimensional shape of the tensor to be filled
  ///  - order: the storage order of the elements
  ///  - name: the name of the tensor
  @inlinable public init(
    ones shape: Shape,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) {
    self.init(shape: shape, order: order, name: name)
    fill(&self, with: 1)
  }

  //--------------------------------------------------------------------------
  /// init(eye:offset:
  /// Returns a new data shape where the bounds and strides are permuted
  /// - Parameters:
  ///  - shape: the shape of the array
  ///  - offset: the offset of the diagonal
  ///  - order: the storage order of the new tensor
  ///  - name: the name of the tensor
  @inlinable public init(
    eye shape: Shape,
    offset: Int = 0,
    order: Order = .defaultOrder,
    name: String = defaultTensorName
  ) {
    self.init(shape: shape, order: order, name: name)
    currentQueue.eye(&self, offset: offset)
  }
}
