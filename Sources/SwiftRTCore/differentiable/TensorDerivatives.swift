//******************************************************************************
// Copyright 2019 Google LLC
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

#if swift(>=5.3) && canImport(_Differentiation)

import Numerics
import _Differentiation

//==============================================================================
/// Derivative registration
extension Tensor where TensorElement.Value: DifferentiableNumeric {
  // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
  
  @derivative(of:subscript)
  @usableFromInline func _derivativeSubscript(
    lower: Shape,
    upper: Shape
  ) -> (value: Self, pullback: (Self) -> Self) {
    return (
      self[lower, upper],
      { v in
        var result = zeros(like: self)
        result[lower, upper] = v
        return result
      }
    )
  }
  
  @derivative(of:element)
  @inlinable public func _derivativeElement() -> (
    value: Element,
    pullback: (Element) -> Self
  ) {
    (
      element,
      { v in
        var result = zeros(like: self)
        result.element = v
        return result
      }
    )
  }
  
  @derivative(of:init(repeating:to:order:name:))
  @usableFromInline static func _vjpInit(
    repeating element: Element,
    to shape: Shape,
    order: Order,
    name: String
  ) -> (value: Self, pullback: (Self) -> (Element)) {
    (
      Self(repeating: element, to: shape, order: order, name: name),
      {
        $0.sum().element
      }
    )
  }

  @derivative(of:init(repeating:to:))
  @usableFromInline static func _vjpInit(repeating other: Self, to shape: Shape)
    -> (value: Self, pullback: (Self) -> (Self))
  {
    // TODO: this is probably wrong. Test this
    (Self(repeating: other, to: shape), { $0 })
  }
  
  @derivative(of:init(reshaping:to:order:))
  @usableFromInline static func _vjpInit<S>(
    reshaping other: Tensor<S, TensorElement>,
    to newShape: Shape,
    order newOrder: Order? = nil
  ) -> (value: Self, pullback: (Self) -> Tensor<S, TensorElement>)
  where S: TensorShape {
    let value = Self(reshaping: other, to: newShape, order: newOrder)
    return (
      value,
      {
        Tensor<S, TensorElement>(
          reshaping: $0, to: other.shape,
          order: other.order)
      }
    )
  }

  @derivative(of:init(expanding:axes:))
  @usableFromInline static func _vjpInit<S, Axes>(
    expanding other: Tensor<S, TensorElement>,
    axes: Axes
  ) -> (value: Self, pullback: (Self) -> Tensor<S, TensorElement>)
  where S: TensorShape, Axes: TensorShape {
    let value = Self(expanding: other, axes: axes)
    return (value, { Tensor<S, TensorElement>(squeezing: $0, axes: axes) })
  }

  @derivative(of:init(squeezing:axes:))
  @usableFromInline static func _vjpInit<S, Axes>(
    squeezing other: Tensor<S, TensorElement>,
    axes: Axes
  ) -> (value: Self, pullback: (Self) -> Tensor<S, TensorElement>)
  where S: TensorShape, Axes: TensorShape {
    let value = Self(squeezing: other, axes: axes)
    return (value, { Tensor<S, TensorElement>(expanding: $0, axes: axes) })
  }

  @derivative(of:init(transposing:permutatedBy:))
  @usableFromInline static func _vjpInit(
    transposing other: Self,
    permutatedBy permutations: Shape?
  ) -> (value: Self, pullback: (Self) -> Self) {
    let value = Self(transposing: other, permutatedBy: permutations)
    return (
      value,
      {
        Self(
          shape: other.shape,
          strides: other.strides,
          count: other.count,
          storage: $0.storage,
          storageBase: $0.storageBase,
          spanCount: other.spanCount,
          order: other.order,
          shared: $0.isShared)
      }
    )
  }
}

@derivative(of:stack)
@inlinable func vjpStack<S, SR, E>(
  _ tensors: [Tensor<S, E>],
  axis: Int = 0,
  into result: inout Tensor<SR, E>
) -> (
  value: (),
  pullback: (inout Tensor<SR, E>.TangentVector)
    -> Array<Tensor<S, E>>.TangentVector
)
where S: TensorShape, SR: TensorShape {
  let tensorCount = tensors.count
  func pullback(_ resultTangent: inout Tensor<SR, E>.TangentVector)
    -> Array<Tensor<S, E>>.TangentVector
  {
    // Fill `tensorTangents` with slices of `resultTangent` of shape
    // `tensorShapes[0]`, `tensorShapes[1]`, etc.
    var tensorTangents: [Tensor<S, E>] = []
    var lower = SR.zero
    var upper = resultTangent.shape
    upper[axis] = 1
    for _ in 0..<tensorCount {
      let slice = Tensor<S, E>(
        squeezing: resultTangent[lower, upper],
        axes: Shape1(axis))
      tensorTangents.append(slice)
      lower[axis] += 1
      upper[axis] += 1
    }

    // Set `resultTangent` to zero.
    // Note: We can't use `fill(_:with:)` because `resultTangent` aliases
    // `tensorTangents`.
    // TODO: track and fix
    // Note: https://bugs.swift.org/browse/TF-1250 will allow us to make
    // this pullback more efficient. How:
    // - Set the wrt parameters and results to
    //     @differentiable(wrt: (tensors), results: (result))
    // - This makes `resultTangent` not be inout, so we don't need to set
    //   it any more.
    resultTangent = zeros(like: resultTangent)

    return Array.DifferentiableView(tensorTangents)
  }
  return (stack(tensors, axis: axis, into: &result), pullback)
}


//==============================================================================
/// DifferentiableTensor
///
/// While these protocols are not strictly necessary, they are used
/// to reduce the number of generic requirements when writing
/// `@differentiable` attributes
///
public protocol TensorProtocol: Logging {
  associatedtype Shape: TensorShape
  associatedtype TensorElement: StorageElement
}

extension Tensor: TensorProtocol { }

public protocol DifferentiableTensor: TensorProtocol & Differentiable
where Self == TangentVector, TensorElement.Value: DifferentiableNumeric {}

/// DifferentiableNumeric
public protocol DifferentiableNumeric:
  Differentiable & Numeric
where Self == TangentVector {}

extension Float: DifferentiableNumeric {}
extension Double: DifferentiableNumeric {}

extension Complex: DifferentiableNumeric
where RealType: Differentiable, RealType.TangentVector == RealType {}

// Differentiable conformance
extension Tensor: Differentiable & DifferentiableTensor
where Element: DifferentiableNumeric {
  public typealias TangentVector = Self

  // This can't be automatically synthesized outside of the file defining Tensor.
  @inlinable
  public var zeroTangentVectorInitializer: () -> Self {
    { Tensor.zero }
  }
}

#endif
