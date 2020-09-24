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
import Numerics

//==============================================================================
// Comparable
// Complex numbers are not technically comparable. However a lexical
// comparison is supported.
// 1) the real component is compared
// 2) if the real components are equal, then the imaginaries are compared
//
extension Complex: Comparable {
    @inlinable public static func < (lhs: Complex<RealType>, rhs: Complex<RealType>) -> Bool {
        if lhs.real == rhs.real {
            return lhs.imaginary < rhs.imaginary
        } else {
            return lhs.real < rhs.real
        }
    }
}

@inlinable public func abs<RealType>(_ x: Complex<RealType>) -> RealType {
    x.imaginary == 0 ? abs(x.real) :
        .sqrt(x.real * x.real + x.imaginary * x.imaginary)
}
