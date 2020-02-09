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

////==============================================================================
///// gradient
///// finds gradients of tensor
//public func gradient<T, R>(
//    at x: T,
//    in fn: @differentiable (T) -> R) -> T.TangentVector
//    where
//    T: DifferentiableTensorView,
//    R: DifferentiableTensorView, R.Shape == T.Shape
//{
//    return pullback(at: x, in: fn)(R(repeating: 1, like: x))
//}
//
//public func gradient<T, U, R>(
//    at x: T, _ y: U,
//    in fn: @differentiable (T, U) -> R) -> (T.TangentVector, U.TangentVector)
//    where
//    T: DifferentiableTensorView,
//    U: DifferentiableTensorView,
//    R: DifferentiableTensorView, R.Shape == T.Shape
//{
//    return pullback(at: x, y, in: fn)(R(repeating: 1, like: x))
//}
//
//public func gradient<T, U, V, R>(
//    at x: T, _ y: U, _ z: V,
//    in fn: @differentiable (T, U, V) -> R) ->
//    (T.TangentVector, U.TangentVector, V.TangentVector)
//    where
//    T: DifferentiableTensorView,
//    U: DifferentiableTensorView,
//    V: DifferentiableTensorView,
//    R: DifferentiableTensorView, R.Shape == T.Shape
//{
//    return pullback(at: x, y, z, in: fn)(R(repeating: 1, like: x))
//}
//
////==============================================================================
///// gradientIsValid
//public func compareGradients<T>(_ grad: T, _ expected: T,
//                                _ tolerance: T.Element) -> Bool
//    where T: DifferentiableTensorView, T.Element: SignedNumeric & Comparable
//{
//    let almostEqual = elementsAlmostEqual(grad, expected, tolerance: tolerance)
//        .all().element
//    if !almostEqual {
//        globalPlatform.current[0].writeLog(
//            "gradient values do not match numerical jvp values")
//        globalPlatform.current[0].writeLog("gradient: \(grad.flatArray)")
//        globalPlatform.current[0].writeLog("expected: \(expected.flatArray)")
//        let maxDiff = (grad - expected).absmax().element
//        globalPlatform.current[0].writeLog("maxDiff: \(maxDiff)")
//    }
//    return almostEqual
//}
//
////==============================================================================
///// gradientIsValid
///// - Parameter at:
//public func gradientIsValid<T>(
//    at x: T,
//    delta: Float = 1e-4,
//    tolerance: Float = 5e-4,
//    in body: @differentiable (T) -> T) -> Bool
//    where T: DifferentiableTensorView, T.Element: AnyFloatingPoint
//{
//    return compareGradients(
//        gradient(at: x, in: body),
//        numericalGradient(at: x, delta: delta, in: body),
//        T.Element(any: tolerance))
//}
//
//public func numericalGradient<T>(
//    at x: T,
//    delta: Float,
//    in body: @differentiable (T) -> T) -> T
//    where T: DifferentiableTensorView, T.Element: AnyFloatingPoint
//{
//    let delta = T.Element(any: delta)
//    let valuePlus = body(x + delta)
//    let valueMinus = body(x - delta)
//    let scale = T.Element(any: 0.5) / delta
//    let diff = valuePlus - valueMinus
//    let result = diff * scale
//    return result
//}
//
////==============================================================================
///// gradientIsValid
///// - Parameter at:
//public func gradientIsValid<T>(
//    at x: T, _ y: T,
//    delta: Float = 1e-4,
//    tolerance: Float = 5e-4,
//    in body: @differentiable (T, T) -> T) -> Bool
//    where T: DifferentiableTensorView, T.Element: AnyFloatingPoint
//{
//    let (dx, dy) = gradient(at: x, y, in: body)
//    let (ndx, ndy) = numericalGradient(at: x, y, delta: delta, in: body)
//    let xEqual = compareGradients(dx, ndx, T.Element(any: tolerance))
//    let yEqual = compareGradients(dy, ndy, T.Element(any: tolerance))
//    return xEqual && yEqual
//}
//
//public func gradientIsValid<T>(
//    at x: T, _ y: T.Element,
//    delta: Float = 1e-4,
//    tolerance: Float = 5e-4,
//    in body: @differentiable (T, T) -> T) -> Bool
//    where T: DifferentiableTensorView, T.Element: AnyFloatingPoint
//{
//    gradientIsValid(at: x, T(repeating: y, like: x),
//                    delta: delta, tolerance: tolerance, in: body)
//}
//
//public func gradientIsValid<T>(
//    at x: T.Element, _ y: T,
//    delta: Float = 1e-4,
//    tolerance: Float = 5e-4,
//    in body: @differentiable (T, T) -> T) -> Bool
//    where T: DifferentiableTensorView, T.Element: AnyFloatingPoint
//{
//    gradientIsValid(at: T(repeating: x, like: y), y,
//                    delta: delta, tolerance: tolerance, in: body)
//}
//
//public func numericalGradient<T>(
//    at x: T, _ y: T,
//    delta: Float,
//    in body: @differentiable (T, T) -> T) -> (T, T)
//    where T: DifferentiableTensorView, T.Element: AnyFloatingPoint
//{
//    let delta = T.Element(any: delta)
//    let scale = T.Element(any: 0.5) / delta
//    let ndx = (body(x + delta, y) - body(x - delta, y)) * scale
//    let ndy = (body(x, y + delta) - body(x, y - delta)) * scale
//    return (ndx, ndy)
//}
//
