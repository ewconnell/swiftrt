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
// PointwiseMultiplicative conformance
extension TensorView where Element: Numeric {
    public static var one: Self {
        fatalError()
    }
    
    public var reciprocal: Self {
        fatalError()
    }
    
    public static func .* (lhs: Self, rhs: Self) -> Self {
        fatalError()
    }
}

//==============================================================================
// ElementaryFunctions conformance
extension TensorView where Self: ElementaryFunctions, Element: Real {
    public static func expMinusOne(_ x: Self) -> Self {
        fatalError()
    }

    public static func sqrt(_ x: Self) -> Self {
        fatalError()
    }
    
    public static func cos(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func sin(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func tan(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func acos(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func asin(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func atan(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func cosh(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func sinh(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func tanh(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func acosh(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func asinh(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func atanh(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func exp(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func exp2(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func exp10(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func expm1(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func log(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func log(onePlus x: Self) -> Self {
        fatalError()
    }
    
    public static func log2(_ x: Self) -> Self {
        fatalError()
    }
    
    public static func log10(_ x: Self) -> Self {
        fatalError()
    }
    
    public static func log1p(_ x: Self) -> Self {
        fatalError()
        
    }
    
    public static func pow(_ x: Self, _ y: Self) -> Self {
        fatalError()
        
    }
    
    public static func pow(_ x: Self, _ n: Int) -> Self {
        fatalError()
        
    }
    
    public static func root(_ x: Self, _ n: Int) -> Self {
        fatalError()
        
    }

}
