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
import Numerics

//==============================================================================
// Platform types
#if canImport(CCuda)
public typealias PlatformServiceType = CudaService
public typealias TensorBuffer<Element> = DiscreetDeviceBuffer<Element>
#else
public typealias PlatformServiceType = CpuService
public typealias TensorBuffer<Element> = CpuBuffer<Element>
#endif

//==============================================================================
// Bound and Shape types
public typealias Bounds1 = SIMD1<Int>
public typealias Bounds2 = SIMD2<Int>
public typealias Bounds3 = SIMD3<Int>
public typealias Bounds4 = SIMD4<Int>
public typealias Bounds5 = SIMD5<Int>

public typealias Shape1 = Shape<SIMD1<Int>>
public typealias Shape2 = Shape<SIMD2<Int>>
public typealias Shape3 = Shape<SIMD3<Int>>
public typealias Shape4 = Shape<SIMD4<Int>>
public typealias Shape5 = Shape<SIMD5<Int>>

//==============================================================================
// Default Tensor types
public typealias IndexType = Int32
public typealias Vector = VectorType<Float>
public typealias BoolVector = VectorType<Bool>
public typealias IndexVector = VectorType<IndexType>
public typealias ComplexVector = VectorType<Complex<Float>>

public typealias Matrix = MatrixType<Float>
public typealias BoolMatrix = MatrixType<Bool>
public typealias IndexMatrix = MatrixType<IndexType>
public typealias ComplexMatrix = MatrixType<Complex<Float>>

public typealias Volume = VolumeType<Float>
public typealias BoolVolume = VolumeType<Bool>
public typealias IndexVolume = VolumeType<IndexType>
public typealias ComplexVolume = VolumeType<Complex<Float>>

//==============================================================================
/// DifferentiableTensorView
///
/// Marker protocol for `TensorView` that conform to `Differentiable`.
///
/// While this protoocl is not strictly necessary, it is used to reduce the
/// number of generic requirements when writing `@differentiable` attributes on
/// generic differentiable `TensorView` functions.
public protocol DifferentiableTensorView: TensorView & Differentiable
    where Self == TangentVector, Element: DifferentiableElement {}

//==============================================================================
/// DifferentiableElement
// this is for shorthand also to make the code less verbose
public protocol DifferentiableElement:
    Differentiable & Numeric where Self == TangentVector {}

extension Float: DifferentiableElement {}
extension Double: DifferentiableElement {}

// this is defined with the typealias because of AD same file
// compiler requirements. Hopefully fixed in the future
extension Complex: DifferentiableElement {
  public typealias TangentVector = Self
}

//==============================================================================
// type extensions
public extension Numeric {
    @inlinable
    static var one: Self { 1 }
}
