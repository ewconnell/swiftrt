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

//==============================================================================
/// TensorIndex
public protocol TensorIndex: Comparable {
    associatedtype Shape: TensorShape
    
    init(_ position: Shape, _ sequenceIndex: Int)
    
    func incremented(boundedBy shape: Shape) -> Self
}

//==============================================================================
/// SeqIndex
/// The sequential index is used to seq
public struct SeqIndex<Shape>: TensorIndex, Codable
    where Shape: TensorShape
{
    /// linear sequence position
    public var sequenceIndex: Int

    /// `init(position:sequenceIndex:`
    @inlinable
    public init(_ position: Shape, _ sequenceIndex: Int) {
        self.sequenceIndex = sequenceIndex
    }
    
    @inlinable public func incremented(boundedBy shape: Shape) -> Self {
        var i = self
        i.sequenceIndex += 1
        return i
    }

    /// `==(lhs:rhs:`
    @inlinable
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.sequenceIndex == rhs.sequenceIndex
    }
    
    /// `<(lhs:rhs`
    @inlinable
    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.sequenceIndex < rhs.sequenceIndex
    }
}

//==============================================================================
/// StridedIndex
/// A strided index is used to iterate through the logical
/// coordinate space specified by `Shape`. A sequence index is also
/// incremented to enable fast index comparison.
public struct StridedIndex<Shape>: TensorIndex, Codable
    where Shape: TensorShape
{
    /// the logical position along each axis
    public var position: Shape
    /// linear sequence position
    public var sequenceIndex: Int

    //------------------------------------
    // initializers
    @inlinable
    public init(_ position: Shape, _ sequenceIndex: Int) {
        self.position = position
        self.sequenceIndex = sequenceIndex
    }

    @inlinable public func incremented(boundedBy shape: Shape) -> Self {
        var i = self
        i.position.increment(boundedBy: shape)
        i.sequenceIndex += 1
        return i
    }

    //------------------------------------
    // Equatable
    @inlinable
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.sequenceIndex == rhs.sequenceIndex
    }
    
    //------------------------------------
    // Comparable
    @inlinable
    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.sequenceIndex < rhs.sequenceIndex
    }
}
