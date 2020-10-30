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
import _Differentiation

// gyb utility docs
// https://nshipster.com/swift-gyb/

//******************************************************************************
//
// DO NOT EDIT. THIS FILE IS GENERATED FROM swift.gyb file
//
//******************************************************************************

/// `Tensor Subscript Behavior`
/// A tensor subscripted with a range returns a sub view.
///
/// A tensor subscripted using `tensor.indices` or an Index formed
/// via the `ElementIndex` structure, will return an `Element`
///
/// A tensor subscripted with integers for each dimension is a convenience
/// function for wrapping the values in an `ElementIndex` structure, and
/// then returning the corresponding tensor `Element` value
///
/// Accessing a collection element value using an integer index calls
/// `read` or `readWrite` to synchronize with the calling thread

//==============================================================================
// Rank1
extension Tensor where Shape == Shape1 {
  /// - Returns: the element
  @inlinable
  public subscript(d0: Int) -> Element {
    get {
      self[makeIndex(at: Shape1(d0))]
    }
    set {
      self[makeIndex(at: Shape1(d0))] = newValue
    }
  }

  /// - Returns: the sub view defined by the range
  @inlinable
  @differentiable(where Element: DifferentiableNumeric)
  public subscript<R0>(r0: R0) -> Self
  where
    R0: SignedRangeExpression
  {
    get {
      let d0 = r0.relativeTo(0..<shape[0])
      let lower = Shape1(d0.lowerBound)
      let upper = Shape1(d0.upperBound)
      return self[lower, upper]
    }

    set {
      let d0 = r0.relativeTo(0..<shape[0])
      let lower = Shape1(d0.lowerBound)
      let upper = Shape1(d0.upperBound)
      self[lower, upper] = newValue
    }
  }
}

//==============================================================================
// Rank2
extension Tensor where Shape == Shape2 {
  /// - Returns: the element
  @inlinable
  public subscript(d0: Int, d1: Int) -> Element {
    get {
      self[makeIndex(at: Shape2(d0, d1))]
    }
    set {
      self[makeIndex(at: Shape2(d0, d1))] = newValue
    }
  }

  /// - Returns: the sub view defined by the range
  @inlinable
  @differentiable(where Element: DifferentiableNumeric)
  public subscript<R0, R1>(r0: R0, r1: R1) -> Self
  where
    R0: SignedRangeExpression,
    R1: SignedRangeExpression
  {
    get {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let lower = Shape2(d0.lowerBound, d1.lowerBound)
      let upper = Shape2(d0.upperBound, d1.upperBound)
      return self[lower, upper]
    }

    set {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let lower = Shape2(d0.lowerBound, d1.lowerBound)
      let upper = Shape2(d0.upperBound, d1.upperBound)
      self[lower, upper] = newValue
    }
  }
}

//==============================================================================
// Rank3
extension Tensor where Shape == Shape3 {
  /// - Returns: the element
  @inlinable
  public subscript(d0: Int, d1: Int, d2: Int) -> Element {
    get {
      self[makeIndex(at: Shape3(d0, d1, d2))]
    }
    set {
      self[makeIndex(at: Shape3(d0, d1, d2))] = newValue
    }
  }

  /// - Returns: the sub view defined by the range
  @inlinable
  @differentiable(where Element: DifferentiableNumeric)
  public subscript<R0, R1, R2>(r0: R0, r1: R1, r2: R2) -> Self
  where
    R0: SignedRangeExpression,
    R1: SignedRangeExpression,
    R2: SignedRangeExpression
  {
    get {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let d2 = r2.relativeTo(0..<shape[2])
      let lower = Shape3(d0.lowerBound, d1.lowerBound, d2.lowerBound)
      let upper = Shape3(d0.upperBound, d1.upperBound, d2.upperBound)
      return self[lower, upper]
    }

    set {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let d2 = r2.relativeTo(0..<shape[2])
      let lower = Shape3(d0.lowerBound, d1.lowerBound, d2.lowerBound)
      let upper = Shape3(d0.upperBound, d1.upperBound, d2.upperBound)
      self[lower, upper] = newValue
    }
  }
}

//==============================================================================
// Rank4
extension Tensor where Shape == Shape4 {
  /// - Returns: the element
  @inlinable
  public subscript(d0: Int, d1: Int, d2: Int, d3: Int) -> Element {
    get {
      self[makeIndex(at: Shape4(d0, d1, d2, d3))]
    }
    set {
      self[makeIndex(at: Shape4(d0, d1, d2, d3))] = newValue
    }
  }

  /// - Returns: the sub view defined by the range
  @inlinable
  @differentiable(where Element: DifferentiableNumeric)
  public subscript<R0, R1, R2, R3>(r0: R0, r1: R1, r2: R2, r3: R3) -> Self
  where
    R0: SignedRangeExpression,
    R1: SignedRangeExpression,
    R2: SignedRangeExpression,
    R3: SignedRangeExpression
  {
    get {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let d2 = r2.relativeTo(0..<shape[2])
      let d3 = r3.relativeTo(0..<shape[3])
      let lower = Shape4(d0.lowerBound, d1.lowerBound, d2.lowerBound, d3.lowerBound)
      let upper = Shape4(d0.upperBound, d1.upperBound, d2.upperBound, d3.upperBound)
      return self[lower, upper]
    }

    set {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let d2 = r2.relativeTo(0..<shape[2])
      let d3 = r3.relativeTo(0..<shape[3])
      let lower = Shape4(d0.lowerBound, d1.lowerBound, d2.lowerBound, d3.lowerBound)
      let upper = Shape4(d0.upperBound, d1.upperBound, d2.upperBound, d3.upperBound)
      self[lower, upper] = newValue
    }
  }
}

//==============================================================================
// Rank5
extension Tensor where Shape == Shape5 {
  /// - Returns: the element
  @inlinable
  public subscript(d0: Int, d1: Int, d2: Int, d3: Int, d4: Int) -> Element {
    get {
      self[makeIndex(at: Shape5(d0, d1, d2, d3, d4))]
    }
    set {
      self[makeIndex(at: Shape5(d0, d1, d2, d3, d4))] = newValue
    }
  }

  /// - Returns: the sub view defined by the range
  @inlinable
  @differentiable(where Element: DifferentiableNumeric)
  public subscript<R0, R1, R2, R3, R4>(r0: R0, r1: R1, r2: R2, r3: R3, r4: R4) -> Self
  where
    R0: SignedRangeExpression,
    R1: SignedRangeExpression,
    R2: SignedRangeExpression,
    R3: SignedRangeExpression,
    R4: SignedRangeExpression
  {
    get {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let d2 = r2.relativeTo(0..<shape[2])
      let d3 = r3.relativeTo(0..<shape[3])
      let d4 = r4.relativeTo(0..<shape[4])
      let lower = Shape5(d0.lowerBound, d1.lowerBound, d2.lowerBound, d3.lowerBound, d4.lowerBound)
      let upper = Shape5(d0.upperBound, d1.upperBound, d2.upperBound, d3.upperBound, d4.upperBound)
      return self[lower, upper]
    }

    set {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let d2 = r2.relativeTo(0..<shape[2])
      let d3 = r3.relativeTo(0..<shape[3])
      let d4 = r4.relativeTo(0..<shape[4])
      let lower = Shape5(d0.lowerBound, d1.lowerBound, d2.lowerBound, d3.lowerBound, d4.lowerBound)
      let upper = Shape5(d0.upperBound, d1.upperBound, d2.upperBound, d3.upperBound, d4.upperBound)
      self[lower, upper] = newValue
    }
  }
}

//==============================================================================
// Rank6
extension Tensor where Shape == Shape6 {
  /// - Returns: the element
  @inlinable
  public subscript(d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int) -> Element {
    get {
      self[makeIndex(at: Shape6(d0, d1, d2, d3, d4, d5))]
    }
    set {
      self[makeIndex(at: Shape6(d0, d1, d2, d3, d4, d5))] = newValue
    }
  }

  /// - Returns: the sub view defined by the range
  @inlinable
  @differentiable(where Element: DifferentiableNumeric)
  public subscript<R0, R1, R2, R3, R4, R5>(r0: R0, r1: R1, r2: R2, r3: R3, r4: R4, r5: R5) -> Self
  where
    R0: SignedRangeExpression,
    R1: SignedRangeExpression,
    R2: SignedRangeExpression,
    R3: SignedRangeExpression,
    R4: SignedRangeExpression,
    R5: SignedRangeExpression
  {
    get {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let d2 = r2.relativeTo(0..<shape[2])
      let d3 = r3.relativeTo(0..<shape[3])
      let d4 = r4.relativeTo(0..<shape[4])
      let d5 = r5.relativeTo(0..<shape[5])
      let lower = Shape6(
        d0.lowerBound, d1.lowerBound, d2.lowerBound, d3.lowerBound, d4.lowerBound, d5.lowerBound)
      let upper = Shape6(
        d0.upperBound, d1.upperBound, d2.upperBound, d3.upperBound, d4.upperBound, d5.upperBound)
      return self[lower, upper]
    }

    set {
      let d0 = r0.relativeTo(0..<shape[0])
      let d1 = r1.relativeTo(0..<shape[1])
      let d2 = r2.relativeTo(0..<shape[2])
      let d3 = r3.relativeTo(0..<shape[3])
      let d4 = r4.relativeTo(0..<shape[4])
      let d5 = r5.relativeTo(0..<shape[5])
      let lower = Shape6(
        d0.lowerBound, d1.lowerBound, d2.lowerBound, d3.lowerBound, d4.lowerBound, d5.lowerBound)
      let upper = Shape6(
        d0.upperBound, d1.upperBound, d2.upperBound, d3.upperBound, d4.upperBound, d5.upperBound)
      self[lower, upper] = newValue
    }
  }
}
