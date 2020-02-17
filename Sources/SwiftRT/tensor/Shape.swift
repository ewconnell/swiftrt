//******************************************************************************
// Copyright 2019 Google LLC
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
/// ShapeIndex
public protocol ShapeIndex: Comparable {
    /// linear data buffer index
    var bufferIndex: Int { get }
    /// the sequential index position
    /// Shapes are iterated from (0, 0, ...) to `extents` along all dimensions
    /// The `sequenceIndex` ranges from 0 to shape.count
    var sequenceIndex: Int { get }
}

public extension ShapeIndex {
    // Equatable
    @inlinable
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.sequenceIndex == rhs.sequenceIndex
    }
    
    // Comparable
    @inlinable
    static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.sequenceIndex < rhs.sequenceIndex
    }
    
    @inlinable
    func distance(to other: Self) -> Int { other.sequenceIndex - sequenceIndex }
}

//==============================================================================
/// VectorShapeIndex
public struct VectorShapeIndex: ShapeIndex, Codable {
    public var sequenceIndex: Int
    public var bufferIndex: Int

    public init() { sequenceIndex = 0; bufferIndex = 0 }
    public init(sequenceIndex: Int, bufferIndex: Int) {
        self.sequenceIndex = sequenceIndex
        self.bufferIndex = bufferIndex
    }
}

//==============================================================================
/// RankedShapeIndex
public struct RankedShapeIndex<Array>: ShapeIndex, Codable
    where Array: StaticArrayProtocol & Codable, Array.Element == Int
{
    public var bufferIndex: Int { stridedPosition.last! }
    public var sequenceIndex: Int
    public var stridedPosition: Array
    
    // initializers
    public init(sequenceIndex: Int, stridedPosition: Array) {
        self.sequenceIndex = sequenceIndex
        self.stridedPosition = stridedPosition
    }
}

//==============================================================================
/// ShapeProtocol
public protocol ShapeProtocol: Codable, Collection {
    // types
    associatedtype Array: StaticArrayProtocol where
        Array: Equatable & Codable,
        Array.Element == Int,
        Array.Index == Int

    /// a rank matched array of zeros
    static var zeros: Array { get }
    /// a rank matched array of ones
    static var ones: Array { get }
    /// a rank matched collection starting index
    static var start: Index { get }

    //--------------------------------------------------------------------------
    // properties
    /// the dense number of elements in the shape
    var count: Int { get }
    /// The strided number of elements spanned by the shape
    var spanCount: Int { get }
    /// The extent of the shape in each dimension
    var extents: Array { get }
    /// The distance to the next element for each dimension
    var strides: Array { get }
        
    //--------------------------------------------------------------------------
    /// Fully specified initializer
    /// - Parameter extents: extent of the shape in each dimension
    /// - Parameter strides: the distance to the next element in each dimension
    init(extents: Array, strides: Array?)
    /// Expanding initializer
    /// - Parameter expanding: the lower order shape to expand
    /// - Parameter axes: the set of axes to be expanded
    init<S>(expanding other: S, alongAxes axes: Set<Int>?) where S: ShapeProtocol
    /// Flattening initializer
    /// - Parameter flattening: the higher order shape to flatten
    init<S>(flattening other: S) where S: ShapeProtocol
    /// Squeezing initializer
    /// - Parameter squeezing: the higher order shape to squeeze
    /// - Parameter axes: the set of axes to be squeezed
    init<S>(squeezing other: S, alongAxes axes: Set<Int>?) where S: ShapeProtocol
}

//==============================================================================
// Collection
extension ShapeProtocol where Index == RankedShapeIndex<Array> {
    // indexing
    @inlinable public var endIndex: Index {
        var end = Self.zeros
        end[0] = spanCount
        return Index(sequenceIndex: count, stridedPosition: end)
    }

    //-----------------------------------
    // Collection
    @inlinable
    public func index(after i: Index) -> Index {
        fatalError()
    }

    @inlinable
    public subscript(index: Index) -> Int {
        fatalError()
    }
}

//==============================================================================
// default implementation
public extension ShapeProtocol {
    //--------------------------------------------------------------------------
    // tuple support
    typealias Tuple = Self.Array.Storage

    @inlinable
    init(extents: Tuple, strides: Tuple? = nil) {
        self.init(extents: Array(extents), strides: Array(strides))
    }
    
    //--------------------------------------------------------------------------
    // computed properties
    /// `true` if the underlying data for the whole shape has a stride of 1.
    @inlinable
    var isContiguous: Bool { count == spanCount }
    /// `true` if the shape has zero elements
    @inlinable
    var isEmpty: Bool { count == 0 }
    /// `true` if the shape has one element
    @inlinable
    var isScalar: Bool { count == 1 }
    /// the index of the last dimension
    @inlinable
    var lastDimension: Int { extents.count - 1 }
    /// the number of sahpe extents
    @inlinable
    var rank: Int { extents.count }
    /// the number of items in extent 0
    @inlinable
    var items: Int { extents[0] }
    /// returns a dense version of self
    @inlinable
    var dense: Self { isContiguous ? self : Self(extents: extents) }
    
    // indexing
    @inlinable var startIndex: Index { Self.start }


    //--------------------------------------------------------------------------
    // computeSpanCount
    // A sub view may cover a wider range of parent element indexes
    // than the number of dense elements defined by the extents of the view
    // due to striding.
    // The span of the extent is the linear index of the last index + 1
    @inlinable
    static func computeSpanCount(_ extents: Array, _ strides: Array) -> Int {
        (zip(extents, strides).reduce(0) { $0 + ($1.0 - 1) * $1.1 }) + 1
    }
    
    //--------------------------------------------------------------------------
    // init(extents:
    @inlinable
    init(extents: Array) { self.init(extents: extents, strides: nil) }

    //--------------------------------------------------------------------------
    // init(expanding:
    @inlinable
    init<S>(expanding other: S, alongAxes axes: Set<Int>? = nil)
        where S: ShapeProtocol
    {
        let rank = Self.zeros.count
        assert(other.rank < rank, "can only expand lower ranked shapes")
        var newExtents = Self.zeros
        var newStrides = Self.zeros
        let axesSet = axes == nil ?
            Set(0..<rank - other.rank) :
            Set(axes!.map { $0 < 0 ? $0 + rank : $0 })
        assert(other.rank + axesSet.count == rank,
               "`other.rank` plus number of specified axes " +
            "must equal the `rank` of this shape")

        var otherAxis = 0
        for i in 0..<rank {
            if axesSet.contains(i) {
                // expanded axes are set to 1
                newExtents[i] = 1
                // strides beyond the other's strides are just 1
                newStrides[i] = otherAxis >= other.rank ? 1 :
                    other.strides[otherAxis]
            } else {
                newExtents[i] = other.extents[otherAxis]
                newStrides[i] = other.strides[otherAxis]
                otherAxis += 1
            }
        }
        self.init(extents: newExtents, strides: newStrides)
    }

    //--------------------------------------------------------------------------
    // init(indenting:
    @inlinable
    init<S>(indenting other: S) where S: ShapeProtocol {
        let rank = Self.zeros.count
        assert(other.rank < rank, "can only expand lower ranked shapes")

        // TODO: find fastest way to copy sequences
        let start = rank - other.rank
        var newExtents = Self.ones
        var newStrides = Self.ones
        for (i, j) in zip(start..<rank, 0..<other.rank) {
            newExtents[i] = other.extents[j]
            newStrides[i] = other.strides[j]
        }
        for i in 0..<start {
            newStrides[i] = other.strides[0]
        }
        self.init(extents: newExtents, strides: newStrides)
    }
    
    //--------------------------------------------------------------------------
    // init(padding:
    @inlinable
    init<S>(padding other: S) where S: ShapeProtocol {
        let rank = Self.zeros.count
        assert(other.rank < rank, "can only expand lower ranked shapes")
        
        // TODO: find fastest way to copy sequences
        var newExtents = Self.ones
        var newStrides = Self.ones
        for i in 0..<other.rank {
            newExtents[i] = other.extents[i]
            newStrides[i] = other.strides[i]
        }
        self.init(extents: newExtents, strides: newStrides)
    }
    
    //--------------------------------------------------------------------------
    // init(squeezing:
    @inlinable
    init<S>(squeezing other: S, alongAxes axes: Set<Int>? = nil)
        where S: ShapeProtocol
    {
        // make sure we have a positive set of axes to squeeze along
        let rank = Self.zeros.count
        var newExtents = Self.zeros
        var newStrides = Self.zeros
        let axesSet = axes == nil ?
            Set(0..<other.rank) :
            Set(axes!.map { $0 < 0 ? other.rank + $0 : $0 })

        var axis = 0
        for otherAxis in 0..<other.rank where
            !(other.extents[otherAxis] == 1 && axesSet.contains(otherAxis))
        {
            assert(axis < rank,
                   "Unsqueezed axes of `other` exceeds rank of this shape")
            newExtents[axis] = other.extents[otherAxis]
            newStrides[axis] = other.strides[otherAxis]
            axis += 1
        }
        self.init(extents: newExtents, strides: newStrides)
    }
    
    //--------------------------------------------------------------------------
    // equal
    @inlinable
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.extents == rhs.extents
    }

    //--------------------------------------------------------------------------
    // denseStrides
    // computes the strides for a dense shape
    @inlinable
    static func denseStrides(_ extents: Array) -> Array {
        var strides = ones
        for i in stride(from: extents.count - 1, through: 1, by: -1) {
            strides[i - 1] = extents[i] * strides[i]
        }
        return strides
    }
        
    //--------------------------------------------------------------------------
    /// joined
    /// - Parameter others: array of data shapes to join
    /// - Parameter axis: the joining axis
    /// - Returns: returns a new shape that is the join with the others
    @inlinable
    func joined(with others: [Self], alongAxis axis: Int) -> Self {
        var newExtents = extents
        newExtents[axis] += others.reduce(0) { $0 + $1.extents[axis] }
        return Self(extents: newExtents)
    }
    
    //--------------------------------------------------------------------------
    /// makePositive(dims:
    /// The user can specify indices from `-rank..<rank`.
    /// Negative numbers reference dimensions from the end of `extents`
    /// This ensures they are resolved to positive values.
    @inlinable
    static func makePositive(dims: Array) -> Array {
        var positive = dims
        for i in 0..<dims.count where positive[i] < 0 {
            positive[i] += dims.count
        }
        return positive
    }

    //--------------------------------------------------------------------------
    /// linearIndex
    ///    returns the linear element index
    @inlinable
    func linearIndex(of index: Array) -> Int {
        let i = zip(index, strides).reduce(0) { $0 + $1.0 * $1.1 }
        assert(i < spanCount)
        return i
    }

    //--------------------------------------------------------------------------
    /// contains
    @inlinable
    func contains(index: Array) -> Bool {
        linearIndex(of: index) <= spanCount
    }
    
    @inlinable
    func contains(other: Self) -> Bool {
        other.spanCount <= spanCount
    }
    
    @inlinable
    func contains(index: Array, extents: Array) -> Bool {
        linearIndex(of: index) +
            Self(extents: extents, strides: strides).spanCount <= spanCount
    }

    //--------------------------------------------------------------------------
    /// columnMajor
    @inlinable
    var columnMajor: Self {
        // return self if already column major
        guard strides[rank-1] < strides[rank-2] else { return self }
        // compute column major strides for the last 2 dimensions
        var cmExtent = extents
        cmExtent.swapAt(rank-1, rank-2)
        var cmStrides = Self.denseStrides(cmExtent)
        cmStrides.swapAt(rank-1, rank-2)
        return Self(extents: extents, strides: cmStrides)
    }
    
    //--------------------------------------------------------------------------
    /// repeated(to repeatedExtents:
    @inlinable
    func repeated(to repeatedExtents: Array) -> Self {
        // make sure the extents are compatible
        assert({
            for i in 0..<rank {
                if extents[i] != 1 && extents[i] != repeatedExtents[i] {
                    return false
                }
            }
            return true
        }(), "repeated tensor extents must be either 1" +
            " or match the repeated tensor extents")

        // compute strides, setting stride to 0 for repeated dimensions
        var repeatedStrides = Self.zeros
        for i in 0..<rank where repeatedExtents[i] == extents[i] {
            repeatedStrides[i] = strides[i]
        }
        return Self(extents: repeatedExtents, strides: repeatedStrides)
    }

    //--------------------------------------------------------------------------
    /// transposed(with permutations:
    /// Returns a new data shape where the extents and strides are permuted
    /// - Parameter permutations: the indice order mapping. `count` must
    ///   equal `rank`
    /// - Returns: transposed/permuted shape
    /// - Precondition: Each value in `permutations` must be in the range
    ///   `-rank..<rank`
    @inlinable
    func transposed(with permutations: Array? = nil) -> Self {
        guard rank > 1 else { return self }
        var newExtents = extents
        var newStrides = strides

        // determine the new extents and strides
        if let perm = permutations {
            let mapping = Self.makePositive(dims: perm)
            for index in 0..<rank {
                newExtents[index] = extents[mapping[index]]
                newStrides[index] = strides[mapping[index]]
            }
        } else {
            // simple swap of last two dimensions
            let r1 = rank-1
            let r2 = rank-2
            newExtents.swapAt(r1, r2)
            newStrides.swapAt(r1, r2)
        }
        return Self(extents: newExtents, strides: newStrides)
    }
}

//==============================================================================
// Shape1
public struct Shape1: ShapeProtocol {
    // types
    public typealias Array = StaticArray<Int, (Int)>
    public typealias Index = VectorShapeIndex
    
    // constants
    public static let zeros = Array((0))
    public static let ones = Array((1))
    public static let start = Index()

    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array
    
    // indexing
    @inlinable public var endIndex: Index {
        Index(sequenceIndex: count, bufferIndex: spanCount)
    }

    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        self.count = extents[0]
        self.spanCount = Self.computeSpanCount(self.extents, self.strides)
    }

    //-----------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        self.init(extents: Array((other.count)))
    }

    //-----------------------------------
    // Collection
    @inlinable
    public func index(after i: Index) -> Index {
        var next = i
        next.sequenceIndex += 1
        next.bufferIndex += strides[0]
        return next
    }

    @inlinable
    public subscript(index: Index) -> Int {
        index.bufferIndex
    }
}

//==============================================================================
// Shape2
public struct Shape2: ShapeProtocol {
    // types
    public typealias Array = StaticArray<Int, (Int, Int)>
    public typealias Index = RankedShapeIndex<Array>
    
    // constants
    public static let zeros = Array((0, 0))
    public static let ones = Array((1, 1))
    public static let start = Index(sequenceIndex: 0, stridedPosition: zeros)

    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array
    
    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        self.count = extents.reduce(1, *)
        self.spanCount = Self.computeSpanCount(self.extents, self.strides)
    }

    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isContiguous, "Cannot flatten strided data")
        assert(other.rank >= 2, "you can't flatten from a lower rank")
        self.init(extents: Array((other.extents[0],
                                  other.count / other.extents[0])))
    }
}

//==============================================================================
// Shape3
public struct Shape3: ShapeProtocol {
    // types
    public typealias Array = StaticArray<Int, (Int, Int, Int)>
    public typealias Index = RankedShapeIndex<Array>

    // constants
    public static let zeros = Array((0, 0, 0))
    public static let ones = Array((1, 1, 1))
    public static let start = Index(sequenceIndex: 0, stridedPosition: zeros)

    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array

    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        count = extents.reduce(1, *)
        spanCount = Self.computeSpanCount(self.extents, self.strides)
    }
    
    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isContiguous, "Cannot flatten strided data")
        assert(other.rank >= 3, "you can't flatten from a lower rank")
        self.init(extents: Array((
            other.extents[0],
            other.extents[1],
            other.extents[2...].reduce(0,+)
        )))
    }
}

//==============================================================================
// Shape4
public struct Shape4: ShapeProtocol {
    // types
    public typealias Array = StaticArray<Int, (Int, Int, Int, Int)>
    public typealias Index = RankedShapeIndex<Array>

    // constants
    public static let zeros = Array((0, 0, 0, 0))
    public static let ones = Array((1, 1, 1, 1))
    public static let start = Index(sequenceIndex: 0, stridedPosition: zeros)

    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array
    
    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        count = extents.reduce(1, *)
        spanCount = Self.computeSpanCount(self.extents, self.strides)
    }
    
    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isContiguous, "Cannot flatten strided data")
        assert(other.rank >= 4, "you can't flatten from a lower rank")
        self.init(extents: Array((
            other.extents[0],
            other.extents[1],
            other.extents[2],
            other.extents[3...].reduce(0,+)
        )))
    }
}

//==============================================================================
// Shape5
public struct Shape5: ShapeProtocol {
    // types
    public typealias Array = StaticArray<Int, (Int, Int, Int, Int, Int)>
    public typealias Index = RankedShapeIndex<Array>

    // constants
    public static let zeros = Array((0, 0, 0, 0, 0))
    public static let ones = Array((1, 1, 1, 1, 1))
    public static let start = Index(sequenceIndex: 0, stridedPosition: zeros)

    // properties
    public let count: Int
    public let spanCount: Int
    public let extents: Array
    public let strides: Array
    
    @inlinable
    public init(extents: Array, strides: Array? = nil) {
        self.extents = extents
        self.strides = strides ?? Self.denseStrides(extents)
        count = extents.reduce(1, *)
        spanCount = Self.computeSpanCount(self.extents, self.strides)
    }
    
    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isContiguous, "Cannot flatten strided data")
        assert(other.rank >= 5, "you can't flatten from a lower rank")
        self.init(extents: Array((
            other.extents[0],
            other.extents[1],
            other.extents[2],
            other.extents[3],
            other.extents[4...].reduce(0,+)
        )))
    }
}
