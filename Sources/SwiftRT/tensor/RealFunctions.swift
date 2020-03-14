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
        Context.platform.acos(x)
    }
    
    public static func acosh(_ x: Self) -> Self {
        Context.platform.acosh(x)
    }

    public static func asin(_ x: Self) -> Self {
        Context.platform.acos(x)
    }
    
    public static func asinh(_ x: Self) -> Self {
        Context.platform.asinh(x)
    }

    public static func atan(_ x: Self) -> Self {
        Context.platform.atan(x)
    }
    
    public static func atanh(_ x: Self) -> Self {
        Context.platform.atanh(x)
    }

    public static func atan2(y: Self, x: Self) -> Self {
        Context.platform.atan2(y: y, x: x)
    }
    
    public static func cos(_ x: Self) -> Self {
        Context.platform.cos(x)
    }
    
    public static func cosh(_ x: Self) -> Self {
        Context.platform.cosh(x)
    }

    public static func erf(_ x: Self) -> Self {
        Context.platform.erf(x)
    }
    
    public static func erfc(_ x: Self) -> Self {
        Context.platform.erfc(x)
    }
    
    public static func exp(_ x: Self) -> Self {
        Context.platform.exp(x)
    }
    
    public static func exp2(_ x: Self) -> Self {
        Context.platform.exp2(x)
    }
    
    public static func exp10(_ x: Self) -> Self {
        Context.platform.exp10(x)
    }

    public static func expMinusOne(_ x: Self) -> Self {
        Context.platform.expMinusOne(x)
    }

    public static func gamma(_ x: Self) -> Self {
        Context.platform.gamma(x)
    }
    
    public static func hypot(_ x: Self, _ y: Self) -> Self {
        Context.platform.hypot(x, y)
    }
    
    public static func logGamma(_ x: Self) -> Self {
        Context.platform.logGamma(x)
    }

    public static func signGamma(_ x: Self) -> FloatingPointSign {
        // TODO: don't know what to do here
        // is this a set operation?
        fatalError()
    }
    
    public static func sin(_ x: Self) -> Self {
        Context.platform.sin(x)
    }
    
    public static func sinh(_ x: Self) -> Self {
        Context.platform.sinh(x)
    }

    public static func sqrt(_ x: Self) -> Self {
        Context.platform.sqrt(x)
    }
    
    public static func tan(_ x: Self) -> Self {
        Context.platform.tan(x)
    }
    
    public static func tanh(_ x: Self) -> Self {
        Context.platform.tanh(x)
    }
    
    public static func log(_ x: Self) -> Self {
        Context.platform.log(x)
    }
    
    public static func log(onePlus x: Self) -> Self {
        Context.platform.log(onePlus: x)
    }
    
    public static func log2(_ x: Self) -> Self {
        Context.platform.log2(x)
    }
    
    public static func log10(_ x: Self) -> Self {
        Context.platform.log10(x)
    }
    
    public static func pow(_ x: Self, _ y: Self) -> Self {
        Context.platform.pow(x, y)
    }
    
    public static func pow(_ x: Self, _ n: Int) -> Self {
        Context.platform.pow(x, n)
    }
    
    public static func root(_ x: Self, _ n: Int) -> Self {
        Context.platform.root(x, n)
    }
}
