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
import SwiftRTCuda
import Numerics

//------------------------------------------------------------------------------
/// withTensorDescriptor
/// creates a SwiftRTCuda compatible tensor descriptor on the stack
/// for use during a driver call
extension Tensor {
    @inlinable public func withTensor<Result>(
        using queue: PlatformType.Device.Queue,
        _ body: (UnsafeRawPointer, UnsafePointer<srtTensorDescriptor>) -> Result
    ) -> Result {
        let deviceDataPointer = deviceRead(using: queue)
        return shape.withUnsafePointer { shapePointer in
            strides.withUnsafePointer { stridesPointer in
                var tensorDescriptor = srtTensorDescriptor(
                    type: TensorElement.type,
                    rank: UInt32(Shape.rank),
                    order: order.cublas,
                    count: count,
                    spanCount: spanCount,
                    shape: shapePointer,
                    strides: stridesPointer
                )

                return withUnsafePointer(to: &tensorDescriptor) {
                    let raw = UnsafeRawPointer($0)
                    return body(
                        deviceDataPointer, 
                        raw.assumingMemoryBound(to: srtTensorDescriptor.self))
                }
            }
        }
    }

    @inlinable public mutating func withMutableTensor<Result>(
        using queue: PlatformType.Device.Queue,
        _ body: (UnsafeMutableRawPointer, UnsafePointer<srtTensorDescriptor>) -> Result
    ) -> Result {
        let deviceDataPointer = deviceReadWrite(using: queue)
        return shape.withUnsafePointer { shapePointer in
            strides.withUnsafePointer { stridesPointer in
                var tensorDescriptor = srtTensorDescriptor(
                    type: TensorElement.type,
                    rank: UInt32(Shape.rank),
                    order: order.cublas,
                    count: count,
                    spanCount: spanCount,
                    shape: shapePointer,
                    strides: stridesPointer
                )

                return withUnsafePointer(to: &tensorDescriptor) {
                    let raw = UnsafeRawPointer($0)
                    return body(
                        deviceDataPointer,
                        raw.assumingMemoryBound(to: srtTensorDescriptor.self))
                }
            }
        }
    }
}

//==============================================================================
// type identifier extensions
extension Bool {
    @inlinable public static var type: srtDataType { boolean }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not implemented") }
    @inlinable public static var cublas: cublasDataType_t { fatalError("not implemented") }
}

extension Bool1 {
    @inlinable public static var type: srtDataType { fatalError("not implemented") }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not implemented") }
    @inlinable public static var cublas: cublasDataType_t { fatalError("not implemented") }
}

extension UInt1 {
    @inlinable public static var type: srtDataType { real1U }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not implemented") }
    @inlinable public static var cublas: cublasDataType_t { fatalError("not implemented") }
}

extension UInt4 {
    @inlinable public static var type: srtDataType { real4U }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not supported") }
    @inlinable public static var cublas: cublasDataType_t { fatalError("not supported") }
}

extension UInt8 {
    @inlinable public static var type: srtDataType { real8U }
    @inlinable public static var cudnn: cudnnDataType_t { CUDNN_DATA_UINT8 }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_8U }
}

extension Int8 {
    @inlinable public static var type: srtDataType { real8I }
    @inlinable public static var cudnn: cudnnDataType_t { CUDNN_DATA_INT8 }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_8I }
}

extension UInt16 {
    @inlinable public static var type: srtDataType { real16U }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not supported") }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_16U }
}

extension Int16 {
    @inlinable public static var type: srtDataType { real16I }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not supported") }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_16I }
}

extension UInt32 {
    @inlinable public static var type: srtDataType { real32U }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not supported") }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_32U }
}

extension Int32 {
    @inlinable public static var type: srtDataType { real32I }
    @inlinable public static var cudnn: cudnnDataType_t { CUDNN_DATA_INT32 }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_32I }
}

extension Float16 {
    @inlinable public static var type: srtDataType { real16F }
    @inlinable public static var cudnn: cudnnDataType_t { CUDNN_DATA_HALF }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_16F }
}

extension BFloat16 {
    @inlinable public static var type: srtDataType { real16BF }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not supported") }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_16BF }
}

extension Float {
    @inlinable public static var type: srtDataType { real32F }
    @inlinable public static var cudnn: cudnnDataType_t { CUDNN_DATA_FLOAT }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_32F }
}

extension Double {
    @inlinable public static var type: srtDataType { real64F }
    @inlinable public static var cudnn: cudnnDataType_t { CUDNN_DATA_DOUBLE }
    @inlinable public static var cublas: cublasDataType_t { CUDA_R_64F }
}

//==============================================================================
// Complex
extension Complex {
    @inlinable public static var type: srtDataType {
        switch RealType.self {
        case is Float.Type: return complex32F
        case is Float16.Type: return complex16F
        case is BFloat16.Type: return complex16BF
        case is Double.Type: return complex64F
        default: fatalError("Complex<\(RealType.self)> not implemented yet")
        }
    }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not supported") }
    @inlinable public static var cublas: cublasDataType_t { cudaDataType(type) }
}

//------------------------------------------------------------------------------
@usableFromInline var _storedZeroComplexFloat = Complex<Float>(0)
@usableFromInline var _storedOneComplexFloat = Complex<Float>(1)

extension Complex: StorageElement where RealType == Float {
    @inlinable public static var storedZeroPointer: UnsafeRawPointer {
        UnsafeRawPointer(&_storedZeroComplexFloat) 
    }
    
    @inlinable public static var storedOnePointer: UnsafeRawPointer {
        UnsafeRawPointer(&_storedOneComplexFloat)
    }
}

//==============================================================================
// RGBA
extension RGBA {
    @inlinable public static var type: srtDataType { fatalError("not implemented") }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not implemented") }
    @inlinable public static var cublas: cublasDataType_t { fatalError("not implemented") }
}

extension Stereo {
    @inlinable public static var type: srtDataType { fatalError("not implemented") }
    @inlinable public static var cudnn: cudnnDataType_t { fatalError("not implemented") }
    @inlinable public static var cublas: cublasDataType_t { fatalError("not implemented") }
}