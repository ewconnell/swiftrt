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
import SwiftRT
import Numerics

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
