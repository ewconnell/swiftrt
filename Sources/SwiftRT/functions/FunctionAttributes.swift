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
import Foundation

//==============================================================================
// BatchNormalizeMode
public enum BatchNormalizeMode: Int, Codable {
    case perActivation
    case spatial
}

//==============================================================================
public enum PoolingMode: Int, Codable {
    case averageExcludePadding
    case averageIncludePadding
    case max
    case maxDeterministic
}

//==============================================================================
public class ActivationInferring<T> where
    T: TensorView, T.Element: FloatingPoint
{
    public func infer(y: inout T, from x: T) throws
    { fatalError("Abstract") }
}

public final class ActivationTraining<T>: ActivationInferring<T> where
    T: TensorView, T.Element: FloatingPoint
{
    public func gradient(y: T, yDiff: T, x: T, xDiff: inout T) throws
    { fatalError("Abstract") }
}

public enum ActivationMode: Int, Codable {
    case sigmoid
    case relu
    case tanh
    case clippedRelu
    case elu
    case identity
}

public extension DeviceQueue {
    func createActivation<T>(
        x: T,
        y: inout T,
        mode: ActivationMode,
        nan: NanPropagation,
        reluCeiling: Double = 0) throws -> ActivationInferring<T>
        where T: TensorView, T.Element: ScalarElement
    {
        fatalError("cpu not implemented")
    }
}

//==============================================================================
public enum TransposeOp: Int, Codable {
    case transpose
    case noTranspose
    case conjugateTranspose
}

//==============================================================================
/// DeviceLimits
/// parameters defining maximum device capabilties
public struct DeviceLimits {
    let maxComputeSharedMemorySize: Int
    let maxComputeWorkGroupCount: (Int, Int, Int)
    let maxComputeWorkGroupInvocations: Int
    let maxComputeWorkGroupSize: (Int, Int, Int)
    let maxMemoryAllocationCount: Int
}

//==============================================================================
public enum SoftmaxAlgorithm: Int, Codable {
    case accurate, fast, log
}

//==============================================================================
public enum SoftmaxMode: Int, Codable {
    case channel, instance
}

