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
public typealias Vector = SwiftRT.Vector<Float>
public typealias BoolVector = SwiftRT.Vector<Bool>
public typealias IndexVector = SwiftRT.Vector<IndexType>
public typealias ComplexVector = SwiftRT.Vector<Complex<Float>>

public typealias Matrix = SwiftRT.Matrix<Float>
public typealias BoolMatrix = SwiftRT.Matrix<Bool>
public typealias IndexMatrix = SwiftRT.Matrix<IndexType>
public typealias ComplexMatrix = SwiftRT.Matrix<Complex<Float>>

public typealias Volume = SwiftRT.Volume<Float>
public typealias BoolVolume = SwiftRT.Volume<Bool>
public typealias IndexVolume = SwiftRT.Volume<IndexType>
public typealias ComplexVolume = SwiftRT.Volume<Complex<Float>>
