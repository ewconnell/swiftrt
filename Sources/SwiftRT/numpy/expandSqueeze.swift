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

// gyb utility docs
// https://nshipster.com/swift-gyb/

//******************************************************************************
//
// DO NOT EDIT. THIS FILE IS GENERATED FROM .swift.gyb file
//
//******************************************************************************

/// expand
/// Expands the shape of a tensor by inserting a new axis that will
/// appear at the axis position in the expanded array shape
/// - Parameters:
///  - dims a: input array
///  - axis: the set of axes to expand in the new shape
///
//==============================================================================
// Rank1
//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axis: Int) -> Tensor2<E> {
    Tensor2<E>(expanding: a, axes: Shape1(axis))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axes: Shape2.Tuple) -> Tensor3<E> {
    Tensor3<E>(expanding: a, axes: Shape2(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axes: Shape3.Tuple) -> Tensor4<E> {
    Tensor4<E>(expanding: a, axes: Shape3(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axes: Shape4.Tuple) -> Tensor5<E> {
    Tensor5<E>(expanding: a, axes: Shape4(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor1<E>, axes: Shape5.Tuple) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape5(axes))
}

//==============================================================================
// Rank2
//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor2<E>, axis: Int) -> Tensor3<E> {
    Tensor3<E>(expanding: a, axes: Shape1(axis))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor2<E>, axes: Shape2.Tuple) -> Tensor4<E> {
    Tensor4<E>(expanding: a, axes: Shape2(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor2<E>, axes: Shape3.Tuple) -> Tensor5<E> {
    Tensor5<E>(expanding: a, axes: Shape3(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor2<E>, axes: Shape4.Tuple) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape4(axes))
}

//==============================================================================
// Rank3
//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor3<E>, axis: Int) -> Tensor4<E> {
    Tensor4<E>(expanding: a, axes: Shape1(axis))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor3<E>, axes: Shape2.Tuple) -> Tensor5<E> {
    Tensor5<E>(expanding: a, axes: Shape2(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor3<E>, axes: Shape3.Tuple) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape3(axes))
}

//==============================================================================
// Rank4
//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor4<E>, axis: Int) -> Tensor5<E> {
    Tensor5<E>(expanding: a, axes: Shape1(axis))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor4<E>, axes: Shape2.Tuple) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape2(axes))
}

//==============================================================================
// Rank5
//@differentiable(where E: DifferentiableElement)
@inlinable public func expand<E>(dims a: Tensor5<E>, axis: Int) -> Tensor6<E> {
    Tensor6<E>(expanding: a, axes: Shape1(axis))
}


/// squeeze
/// Remove length one entries from the shape of a tensor
/// - Parameters:
///  - a: input array
///  - axis: the set of axes to squeeze in the shape
///
//==============================================================================
// Rank2
//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor2<E>, axis: Int) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape1(axis))
}

//==============================================================================
// Rank3
//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor3<E>, axis: Int) -> Tensor2<E> {
    Tensor2<E>(squeezing: a, axes: Shape1(axis))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor3<E>, axes: Shape2.Tuple) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape2(axes))
}

//==============================================================================
// Rank4
//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor4<E>, axis: Int) -> Tensor3<E> {
    Tensor3<E>(squeezing: a, axes: Shape1(axis))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor4<E>, axes: Shape2.Tuple) -> Tensor2<E> {
    Tensor2<E>(squeezing: a, axes: Shape2(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor4<E>, axes: Shape3.Tuple) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape3(axes))
}

//==============================================================================
// Rank5
//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor5<E>, axis: Int) -> Tensor4<E> {
    Tensor4<E>(squeezing: a, axes: Shape1(axis))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor5<E>, axes: Shape2.Tuple) -> Tensor3<E> {
    Tensor3<E>(squeezing: a, axes: Shape2(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor5<E>, axes: Shape3.Tuple) -> Tensor2<E> {
    Tensor2<E>(squeezing: a, axes: Shape3(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor5<E>, axes: Shape4.Tuple) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape4(axes))
}

//==============================================================================
// Rank6
//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axis: Int) -> Tensor5<E> {
    Tensor5<E>(squeezing: a, axes: Shape1(axis))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axes: Shape2.Tuple) -> Tensor4<E> {
    Tensor4<E>(squeezing: a, axes: Shape2(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axes: Shape3.Tuple) -> Tensor3<E> {
    Tensor3<E>(squeezing: a, axes: Shape3(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axes: Shape4.Tuple) -> Tensor2<E> {
    Tensor2<E>(squeezing: a, axes: Shape4(axes))
}

//@differentiable(where E: DifferentiableElement)
@inlinable public func squeeze<E>(_ a: Tensor6<E>, axes: Shape5.Tuple) -> Tensor1<E> {
    Tensor1<E>(squeezing: a, axes: Shape5(axes))
}

