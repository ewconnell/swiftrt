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
// https://medium.com/intuitionmachine/should-deep-learning-use-complex-numbers-edbd3aac3fb8
import Complex

//==============================================================================
// swift build -Xswiftc -Xllvm -Xswiftc -enable-experimental-cross-file-derivative-registration
// TODO uncomment when AD same file requirement is lifted
//extension Complex where
//    RealType: Differentiable,
//    RealType.TangentVector == RealType
//{
//    @inlinable
//    @derivative(of: init(_:_:))
//    static func _vjpInit(real: RealType, imaginary: RealType) ->
//        (value: Complex, pullback: (Complex) -> (RealType, RealType))
//    {
//        (Complex(real, imaginary), { ($0.real, $0.imaginary) })
//    }
//}

//==============================================================================
// 
