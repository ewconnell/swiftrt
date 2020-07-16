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
// matmulGradients
// _vjpMatmul helper function
@inlinable public func matmulGradients<E>(
    _ out: TensorR2<E>,
    _ lhs: TensorR2<E>, _ transposeLhs: Bool,
    _ rhs: TensorR2<E>, _ transposeRhs: Bool
) -> (TensorR2<E>, TensorR2<E>)
where E: StorageElement, E.Value: ScalarElement & DifferentiableElement
{
    let (lhsGrad, rhsGrad): (TensorR2<E>, TensorR2<E>)
    switch (transposeLhs, transposeRhs) {
    case (false, false):
        lhsGrad = matmul(out, transposed: false, rhs, transposed: true)
        rhsGrad = matmul(lhs, transposed: true, out, transposed: false)
    case (false, true):
        lhsGrad = matmul(out, rhs)
        rhsGrad = matmul(lhs, transposed: true, out, transposed: false)
    case (true, false):
        lhsGrad = matmul(out, transposed: false, rhs, transposed: true)
        rhsGrad = matmul(lhs, out)
    case (true, true):
        lhsGrad = matmul(out, transposed: true, rhs, transposed: true)
        rhsGrad = matmul(lhs, transposed: true, out, transposed: true)
    }
    return (lhsGrad, rhsGrad)
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
@differentiable(where E.Value: DifferentiableElement)
@differentiable(wrt: lhs where E.Value: DifferentiableElement)
@differentiable(wrt: rhs where E.Value: DifferentiableElement)
@inlinable public func matmul<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> TensorR2<E> where E: StorageElement, E.Value: ScalarElement & Numeric
{
    let lhsShape = transposeLhs ? lhs.shape.t : lhs.shape
    let rhsShape = transposeRhs ? rhs.shape.t : rhs.shape
    assert(lhsShape[1] == rhsShape[0], "matmul inner dimensions must be equal")
    var result = TensorR2<E>(shape: Shape2(lhsShape[0], rhsShape[1]),
                             order: lhs.order)
    let op = Context.currentQueue.matmul2(type: E.self)
    op.forward(lhs, transposeLhs, rhs, transposeRhs, &result)
    return result
}

@derivative(of: matmul)
@inlinable public func _vjpMatmul<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>, TensorR2<E>))
where E: StorageElement, E.Value: ScalarElement & DifferentiableElement
{
    (matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
     { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs) })
}

@derivative(of: matmul, wrt: lhs)
@inlinable public func _vjpMatmulWrtLhs<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>))
where E: StorageElement, E.Value: ScalarElement & DifferentiableElement
{
    (matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
     { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs).0 })
}

@derivative(of: matmul, wrt: rhs)
@inlinable public func _vjpMatmulWrtRhs<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>))
where E: StorageElement, E.Value: ScalarElement & DifferentiableElement
{
    (matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
     { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs).1 })
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
@differentiable(where E.Value: DifferentiableElement)
@differentiable(wrt: (lhs, bias) where E.Value: DifferentiableElement)
@differentiable(wrt: (rhs, bias) where E.Value: DifferentiableElement)
@inlinable public func matmul<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false,
    bias: TensorR1<E>
) -> TensorR2<E> where E: StorageElement, E.Value: ScalarElement & Numeric {
    let lhsShape = transposeLhs ? lhs.shape.t : lhs.shape
    let rhsShape = transposeRhs ? rhs.shape.t : rhs.shape
    assert(lhsShape[1] == rhsShape[0], "matmul inner dimensions must be equal")
    var result = TensorR2<E>(shape: Shape2(lhsShape[0], rhsShape[1]),
                             order: lhs.order)
    let op = Context.currentQueue.matmul2(type: E.self)
    op.forward(lhs, transposeLhs, rhs, transposeRhs, &result)
    return result
}

@derivative(of: matmul)
@inlinable public func _vjpMatmul<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false,
    bias: TensorR1<E>
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>, TensorR2<E>, TensorR1<E>))
where E: StorageElement, E.Value: ScalarElement & DifferentiableElement
{
    fatalError()
//    (matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
//     { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs) })
}

@derivative(of: matmul, wrt: (lhs, bias))
@inlinable public func _vjpMatmulWrtLhsBias<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false,
    bias: TensorR1<E>
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>, TensorR1<E>))
where E: StorageElement, E.Value: ScalarElement & DifferentiableElement
{
    fatalError()
//    (matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
//     { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs) })
}

@derivative(of: matmul, wrt: (rhs, bias))
@inlinable public func _vjpMatmulWrtRhsBias<E>(
    _ lhs: TensorR2<E>, transposed transposeLhs: Bool = false,
    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false,
    bias: TensorR1<E>
) -> (value: TensorR2<E>, pullback: (TensorR2<E>) -> (TensorR2<E>, TensorR1<E>))
where E: StorageElement, E.Value: ScalarElement & DifferentiableElement
{
    fatalError()
//    (matmul(lhs, transposed: transposeLhs, rhs, transposed: transposeRhs),
//     { matmulGradients($0, lhs, transposeLhs, rhs, transposeRhs) })
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
//@differentiable(where E.Value: DifferentiableElement)
//@inlinable public func matmul<S,E>(
//    _ lhs: Tensor<S,E>, transposed transposeLhs: Bool = false,
//    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
//) -> Tensor<S,E> where S: TensorShape, E.Value: Numeric
//{
//    fatalError()
//}
//
//@derivative(of: matmul)
//@inlinable public func _vjpMatmul<S,E>(
//    _ lhs: Tensor<S,E>, transposed transposeLhs: Bool = false,
//    _ rhs: TensorR2<E>, transposed transposeRhs: Bool = false
//) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, TensorR2<E>))
//where S: TensorShape, E.Value: DifferentiableElement
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
//@differentiable(where E.Value: DifferentiableElement)
//@inlinable public func matmul<S,E>(
//    _ lhs: TensorR2<E>, transposed transposeRhs: Bool = false,
//    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
//) -> Tensor<S,E> where S: TensorShape, E.Value: Numeric
//{
//    fatalError()
//}
//
//@derivative(of: matmul)
//@inlinable public func _vjpMatmul<S,E>(
//    _ lhs: TensorR2<E>, transposed transposeRhs: Bool = false,
//    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
//) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (TensorR2<E>, Tensor<S,E>))
//where S: TensorShape, E.Value: DifferentiableElement
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
//@differentiable(where E.Value: DifferentiableElement)
//@inlinable public func matmul<S,E>(
//    _ lhs: Tensor<S,E>, transposed transposeRhs: Bool = false,
//    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
//) -> Tensor<S,E> where S: TensorShape, E.Value: Numeric
//{
//    fatalError()
//}
//
//@derivative(of: matmul)
//@inlinable public func _vjpMatmul<S,E>(
//    _ lhs: Tensor<S,E>, transposed transposeRhs: Bool = false,
//    _ rhs: Tensor<S,E>, transposed transposeLhs: Bool = false
//) -> (value: Tensor<S,E>, pullback: (Tensor<S,E>) -> (Tensor<S,E>, Tensor<S,E>))
//where S: TensorShape, E.Value: DifferentiableElement
//{
//    fatalError()
//}
