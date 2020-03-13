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
// RealFunctions and ElementaryFunctions conformance
// which delegates to the platform
extension TensorView
    where Self: RealFunctions & ElementaryFunctions, Element: Real
{
    public static func acos(_ x: Self) -> Self {
        Platform.service.acos(x)
    }
    
    public static func acosh(_ x: Self) -> Self {
        Platform.service.acosh(x)
    }

    public static func asin(_ x: Self) -> Self {
        Platform.service.acos(x)
    }
    
    public static func asinh(_ x: Self) -> Self {
        Platform.service.asinh(x)
    }

    public static func atan(_ x: Self) -> Self {
        Platform.service.atan(x)
    }
    
    public static func atanh(_ x: Self) -> Self {
        Platform.service.atanh(x)
    }

    public static func atan2(y: Self, x: Self) -> Self {
        Platform.service.atan2(y: y, x: x)
    }
    
    public static func cos(_ x: Self) -> Self {
        Platform.service.cos(x)
    }
    
    public static func cosh(_ x: Self) -> Self {
        Platform.service.cosh(x)
    }

    public static func erf(_ x: Self) -> Self {
        Platform.service.erf(x)
    }
    
    public static func erfc(_ x: Self) -> Self {
        Platform.service.erfc(x)
    }
    
    public static func exp(_ x: Self) -> Self {
        Platform.service.exp(x)
    }
    
    public static func exp2(_ x: Self) -> Self {
        Platform.service.exp2(x)
    }
    
    public static func exp10(_ x: Self) -> Self {
        Platform.service.exp10(x)
    }

    public static func expMinusOne(_ x: Self) -> Self {
        Platform.service.expMinusOne(x)
    }

    public static func gamma(_ x: Self) -> Self {
        Platform.service.gamma(x)
    }
    
    public static func hypot(_ x: Self, _ y: Self) -> Self {
        Platform.service.hypot(x, y)
    }
    
    public static func logGamma(_ x: Self) -> Self {
        Platform.service.logGamma(x)
    }

    public static func signGamma(_ x: Self) -> FloatingPointSign {
        // TODO: don't know what to do here
        // is this a set operation?
        fatalError()
    }
    
    public static func sin(_ x: Self) -> Self {
        Platform.service.sin(x)
    }
    
    public static func sinh(_ x: Self) -> Self {
        Platform.service.sinh(x)
    }

    public static func sqrt(_ x: Self) -> Self {
        Platform.service.sqrt(x)
    }
    
    public static func tan(_ x: Self) -> Self {
        Platform.service.tan(x)
    }
    
    public static func tanh(_ x: Self) -> Self {
        Platform.service.tanh(x)
    }
    
    public static func log(_ x: Self) -> Self {
        Platform.service.log(x)
    }
    
    public static func log(onePlus x: Self) -> Self {
        Platform.service.log(onePlus: x)
    }
    
    public static func log2(_ x: Self) -> Self {
        Platform.service.log2(x)
    }
    
    public static func log10(_ x: Self) -> Self {
        Platform.service.log10(x)
    }
    
    public static func pow(_ x: Self, _ y: Self) -> Self {
        Platform.service.pow(x, y)
    }
    
    public static func pow(_ x: Self, _ n: Int) -> Self {
        Platform.service.pow(x, n)
    }
    
    public static func root(_ x: Self, _ n: Int) -> Self {
        Platform.service.root(x, n)
    }
}
