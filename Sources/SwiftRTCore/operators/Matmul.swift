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
/// matmul
/// performs a matrix cross product
/// - Parameters:
///  - lhs: left hand tensor
///  - transposeLhs: `true` to transpose `lhs`, default is `false`
///  - rhs: right hand tensor.
///  - transposeRhs: `true` to transpose `rhs`, default is `false`
/// - Returns: a new tensor containing the result
// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
@inlinable public func matmul<E>(
  _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
  _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> TensorR2<E> where E: StorageElement, E.Value: StorageElement & Numeric {
  let lhsShape = transposeLhs ? lhs.shape.t : lhs.shape
  let rhsShape = transposeRhs ? rhs.shape.t : rhs.shape
  assert(lhsShape[1] == rhsShape[0], "matmul inner dimensions must be equal")
  var result = TensorR2<E>(
    shape: Shape2(lhsShape[0], rhsShape[1]),
    order: lhs.order)

  currentQueue.matmul(lhs, transposeLhs, rhs, transposeRhs, &result)

  // let op = currentQueue.matmul2(type: E.self)
  // op.forward(lhs, transposeLhs, rhs, transposeRhs, &result)
  return result
}

//==============================================================================
/// matmul
/// performs a matrix cross product
/// - Parameters:
///  - lhs: left hand tensor
///  - transposeLhs: `true` to transpose `lhs`, default is `false`
///  - rhs: right hand tensor.
///  - transposeRhs: `true` to transpose `rhs`, default is `false`
/// - Returns: a new tensor containing the result
// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
@inlinable public func matmul<E>(
  _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
  _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false,
  bias: TensorR1<E>
) -> TensorR2<E> where E: StorageElement, E.Value: StorageElement & Numeric {
  let lhsShape = transposeLhs ? lhs.shape.t : lhs.shape
  let rhsShape = transposeRhs ? rhs.shape.t : rhs.shape
  assert(lhsShape[1] == rhsShape[0], "matmul inner dimensions must be equal")
  var result = TensorR2<E>(
    shape: Shape2(lhsShape[0], rhsShape[1]),
    order: lhs.order)
  //    let op = currentQueue.matmul2(type: E.self)
  //    op.forward(lhs, transposeLhs, rhs, transposeRhs, &result)
  currentQueue.matmul(lhs, transposeLhs, rhs, transposeRhs, &result)
  return result
}

////==============================================================================
///// matmul
///// performs a batched matrix cross product
///// - Parameters:
/////  - lhs: left hand batched tensor
/////  - transposeLhs: `true` to transpose `lhs`, default is `false`
/////  - rhs: right hand 2D tensor
/////  - transposeRhs: `true` to transpose `rhs`, default is `false`
///// - Returns: a new tensor containing the result
//// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
//@differentiable(where E.Value: DifferentiableNumeric)
//@inlinable public func matmul<S,E>(
//    _ lhs: Tensor<S,E>, transposed transposeLhs: Bool = false,
//    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
//) -> Tensor<S,E> where S: TensorShape, E.Value: Numeric
//{
//    fatalError()
//}
//
////==============================================================================
///// matmul
///// performs a batched matrix cross product
///// - Parameters:
/////  - lhs: left hand batched tensor
/////  - transposeLhs: `true` to transpose `lhs`, default is `false`
/////  - rhs: right hand 2D tensor
/////  - transposeRhs: `true` to transpose `rhs`, default is `false`
///// - Returns: a new tensor containing the result
//// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
//@differentiable(where E.Value: DifferentiableNumeric)
//@inlinable public func matmul<S,E>(
//    _ lhs: TensorR2<E>, transposed transposeRhs: Bool = false,
//    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
//) -> Tensor<S,E> where S: TensorShape, E.Value: Numeric
//{
//    fatalError()
//}
//
////==============================================================================
///// matmul
///// performs a batched matrix cross product
///// - Parameters:
/////  - lhs: left hand batched tensor
/////  - transposeLhs: `true` to transpose `lhs`, default is `false`
/////  - rhs: right hand batched tensor
/////  - transposeRhs: `true` to transpose `rhs`, default is `false`
///// - Returns: a new tensor containing the result
//// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
//@differentiable(where E.Value: DifferentiableNumeric)
//@inlinable public func matmul<S,E>(
//    _ lhs: Tensor<S,E>, transposed transposeRhs: Bool = false,
//    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
//) -> Tensor<S,E> where S: TensorShape, E.Value: Numeric
//{
//    fatalError()
//}
