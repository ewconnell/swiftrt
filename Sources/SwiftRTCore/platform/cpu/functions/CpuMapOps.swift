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

extension DeviceQueue {
    //--------------------------------------------------------------------------
    @inlinable func generatorOp<R>(
        _ r: inout R,
        _ op: @escaping () -> R.Element
    ) where R: MutableCollection {
        r.indices.forEach { r[$0] = op() }
    }
    
    //--------------------------------------------------------------------------
    @inlinable func inPlaceOp<R>(
        _ r: inout R,
        _ op: @escaping (R.Element) -> R.Element
    ) where R: MutableCollection {
        r.indices.forEach { r[$0] = op(r[$0]) }
    }
    
    //--------------------------------------------------------------------------
    // mapOp 1
    @inlinable func mapOp<T, R>(
        _ x: T,
        _ r: inout R,
        _ op: @escaping (T.Element) -> R.Element
    ) where T: Collection, R: MutableCollection {
        zip(r.indices, x).forEach { r[$0] = op($1) }
    }
    
    //--------------------------------------------------------------------------
    // mapOp 2
    @inlinable func mapOp<S,E,RE>(
        _ lhs: Tensor<S,E>,
        _ rhs: Tensor<S,E>,
        _ r: inout Tensor<S,RE>,
        _ op: @escaping (E, E) -> RE
    ) {
        lhs.read(using: self)
        rhs.read(using: self)
        r.readWrite(using: self)
        zip(r.indices, zip(lhs, rhs)).forEach { r[$0] = op($1.0, $1.1) }
    }

    //--------------------------------------------------------------------------
    // mapOp 3
    @inlinable
    func mapOp<T1, T2, T3, R>(
        _ a: T1,
        _ b: T2,
        _ c: T3,
        _ r: inout R,
        _ op: @escaping (T1.Element, T2.Element, T3.Element) -> R.Element
    ) where T1: Collection, T2: Collection, T3: Collection, R: MutableCollection {
        zip(r.indices, zip(a, zip(b, c))).forEach { r[$0] = op($1.0, $1.1.0, $1.1.1) }
    }
    
    //--------------------------------------------------------------------------
    // mapOp 3R2
    /// generically combines three tensors
    @inlinable func mapOp<T1, T2, T3, R1, R2>(
        _ a: T1,
        _ b: T2,
        _ c: T3,
        _ r1: inout R1,
        _ r2: inout R2,
        _ op: @escaping
            (T1.Element, T2.Element, T3.Element) -> (R1.Element, R2.Element)
    ) where T1: Collection, T2: Collection, T3: Collection,
            R1: MutableCollection, R2: MutableCollection
    {
        zip(zip(r1.indices, r2.indices), zip(a, zip(b, c))).forEach {
            let (r1v, r2v) = op($1.0, $1.1.0, $1.1.1)
            r1[$0.0] = r1v
            r2[$0.1] = r2v
        }
    }
    
    //--------------------------------------------------------------------------
    // reductionOp
    @inlinable func reductionOp<T, R>(
        _ x: T, _ r: inout R,
        _ op: @escaping (T.Element, T.Element) -> T.Element
    ) where T: Collection, R: MutableCollection, R.Element == T.Element {
        zip(r.indices, x).forEach { r[$0] = op(r[$0], $1) }
    }
}
