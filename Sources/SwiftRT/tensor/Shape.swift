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
/// ShapeProtocol
public protocol ShapeProtocol: Codable, Equatable, Collection
    where Element == Int
{
    associatedtype Bounds: ShapeBounds
    
    // properties
    /// the dense number of elements in the shape
    var count: Int { get }
    /// the bounds of the shape in each dimension
    var bounds: Bounds { get }
    /// `true` if indexing is row sequential for performance
    var isSequential: Bool { get }
    /// The strided number of elements spanned by the shape
    var spanCount: Int { get }
    /// Specifies whether data is stored in row-major (C-style)
    /// or column-major (Fortran-style) order in memory.
    var order: StorageOrder { get }
    /// The distance to the next element for each dimension
    var strides: Bounds { get }
        
    /// init(bounds:strides:order:
    /// - Parameters:
    ///  - bounds: bounds of the shape in each dimension
    ///  - strides: the distance to the next element in each dimension
    ///  - order: specifies whether data is stored in row-major (C-style)
    ///    or column-major (Fortran-style) order in memory.
    init(bounds: Bounds, strides: Bounds?, storage order: StorageOrder)
}

//==============================================================================
// messages
@usableFromInline
let _messageInvalidBounds = "bounding dimensions must be greater than 0"

//==============================================================================
// ShapeProtocol extensions
extension ShapeProtocol {
    /// array
    @inlinable
    public var array: [Int] { [Int](self) }
    /// `true` if the shape has zero elements
    @inlinable
    public var isEmpty: Bool { count == 0 }
    /// `true` if the shape has one element
    @inlinable
    public var isScalar: Bool { count == 1 }
    /// the number of items in extent 0
    @inlinable
    public var items: Int { bounds[0] }
    /// returns a dense version of self
    @inlinable
    public var dense: Self { isSequential ? self : Self(bounds) }
    /// the static rank of the shape
    @inlinable @_transparent
    public static var rank: Int { Bounds.rank }

    //--------------------------------------------------------------------------
    // getSpanCount
    // A sub view may cover a wider range of parent element indexes
    // than the number of dense elements defined by the bounds of the view
    // due to striding.
    // The span of the bounds is the linear index of the last index + 1
    @inlinable
    public static func spanCount(for bounds: Bounds, with strides: Bounds)-> Int
    {
        ((bounds &- 1) &* strides).wrappedSum() + 1
    }

    //--------------------------------------------------------------------------
    /// `elementCount`
    /// the count of logical elements described by bounds
    @inlinable
    public static func elementCount(of bounds: Bounds) -> Int {
        bounds.indices.reduce(into: 1) { $0 &*= bounds[$1] }
    }

    //--------------------------------------------------------------------------
    /// `sequentialStrides(bounds:`
    /// - Returns: the row major sequential strides for the given bounds
    @inlinable
    public static func sequentialStrides(for bounds: Bounds) -> Bounds {
        var strides = Bounds.one
        for i in stride(from: Self.rank - 1, through: 1, by: -1) {
            strides[i - 1] = bounds[i] * strides[i]
        }
        return strides
    }
    
    //--------------------------------------------------------------------------
    // linearIndex
    @inlinable
    public func linearIndex(of position: Bounds) -> Int {
        (position &* strides).wrappedSum()
    }

    //--------------------------------------------------------------------------
    // init(bounds:order:
    @inlinable
    public init(_ bounds: Bounds, storage order: StorageOrder = .C) {
        self.init(bounds: bounds,
                  strides: Self.sequentialStrides(for: bounds),
                  storage: order)
    }
    
    //--------------------------------------------------------------------------
    // init(bounds:order:
    @inlinable
    public init(_ bounds: Bounds, strides: Bounds,
                storage order: StorageOrder = .C)
    {
        self.init(bounds: bounds, strides: strides, storage: order)
    }

    //--------------------------------------------------------------------------
    // init(expanding:
    @inlinable
    public init<S>(expanding other: S, alongAxes axes: Set<Int>? = nil)
        where S: ShapeProtocol
    {
        assert(S.rank < Self.rank, "can only expand lower ranked shapes")
        var newBounds = Bounds.zero
        var newStrides = Bounds.zero
        let axesSet = axes == nil ?
            Set(0..<Self.rank - S.rank) :
            Set(axes!.map { $0 < 0 ? $0 + Self.rank : $0 })
        assert(S.rank + axesSet.count == Self.rank,
               "`other.rank` plus number of specified axes " +
            "must equal the `rank` of this shape")

        var j = S.rank - 1
        for i in (0..<Self.rank).reversed() {
            if axesSet.contains(i) {
                // expanded axes are set to 1
                newBounds[i] = 1
                // repeat stride of next dimension or pad with 1
                if i == Self.rank - 1 {
                    newStrides[i] = 1
                } else {
                    newStrides[i] = newBounds[i + 1] * newStrides[i + 1]
                }
            } else {
                newBounds[i] = other.bounds[j]
                newStrides[i] = other.strides[j]
                j -= 1
            }
        }
        self.init(newBounds, strides: newStrides, storage: other.order)
    }
    
    //--------------------------------------------------------------------------
    // init(indenting:
    @inlinable
    public init<S>(indenting other: S) where S: ShapeProtocol {
        assert(S.rank < Self.rank, "can only indent lower ranked shapes")

        // Self and other are different ranks so we append other's elements
        let start = Self.rank - S.rank
        var newBounds = Bounds.one
        var newStrides = Bounds.one
        for (i, j) in zip(start..<Self.rank, 0..<S.rank) {
            newBounds[i] = other.bounds[j]
            newStrides[i] = other.strides[j]
        }
        for i in 0..<start {
            newStrides[i] = other.strides[0]
        }
        
        self.init(newBounds, strides: newStrides, storage: other.order)
    }
    
    //--------------------------------------------------------------------------
    // init(padding:
    @inlinable
    public init<S>(padding other: S) where S: ShapeProtocol {
        assert(S.rank < Self.rank, "can only pad lower ranked shapes")
        
        // Self and other are different ranks so we copy the leading elements
        var newBounds = Bounds.one
        var newStrides = Bounds.one
        for i in 0..<S.rank {
            newBounds[i] = other.bounds[i]
            newStrides[i] = other.strides[i]
        }
        self.init(newBounds, strides: newStrides, storage: other.order)
    }
    
    //--------------------------------------------------------------------------
    // init(squeezing:
    @inlinable
    public init<S>(squeezing other: S, alongAxes axes: Set<Int>? = nil)
        where S: ShapeProtocol
    {
        // make sure we have a positive set of axes to squeeze along
        var newBounds = Bounds.zero
        var newStrides = Bounds.zero
        let axesSet = axes == nil ?
            Set(0..<S.rank) :
            Set(axes!.map { $0 < 0 ? S.rank + $0 : $0 })

        var axis = 0
        for otherAxis in 0..<S.rank where
            !(other.bounds[otherAxis] == 1 && axesSet.contains(otherAxis))
        {
            assert(axis < Self.rank,
                   "Unsqueezed axes of `other` exceeds rank of this shape")
            newBounds[axis] = other.bounds[otherAxis]
            newStrides[axis] = other.strides[otherAxis]
            axis += 1
        }
        self.init(newBounds, strides: newStrides, storage: other.order)
    }
    
    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isSequential, "cannot flatten non sequential data")
        assert(S.rank >= Self.rank, "cannot flatten bounds of lower rank")

        // copy the leading dimensions
        var bounds = Bounds.zero
        for i in 0..<Self.rank {
            bounds[i] = other.bounds[i]
        }

        // get product of the remaining dimensions
        for j in Self.rank..<S.rank {
            bounds[Self.rank-1] *= other.bounds[j]
        }
        self = Self(bounds)
    }
    
    //--------------------------------------------------------------------------
    /// joined
    /// - Parameter others: array of data shapes to join
    /// - Parameter axis: the joining axis
    /// - Returns: returns a new shape that is the join with the others
    @inlinable
    public func joined(with others: [Self], alongAxis axis: Int) -> Self {
        var newBounds = bounds
        newBounds[axis] += others.reduce(into: 0) { $0 += $1.bounds[axis] }
        return Self(newBounds)
    }
    
    //--------------------------------------------------------------------------
    /// makePositive(bounds:
    /// The user can specify indices from `-rank..<rank`.
    /// Negative numbers reference dimensions from the end of `bounds`
    /// This ensures they are resolved to positive values.
    @inlinable
    public static func makePositive(bounds: Bounds) -> Bounds {
        var positive = bounds
        for i in 0..<Bounds.rank where positive[i] < 0 {
            positive[i] += Bounds.rank
        }
        return positive
    }

    //--------------------------------------------------------------------------
    /// contains
    @inlinable
    public func contains(_ point: Bounds) -> Bool {
        linearIndex(of: point) <= spanCount
    }

    @inlinable
    public func contains(other: Self) -> Bool {
        other.spanCount <= spanCount
    }

    //--------------------------------------------------------------------------
    /// columnMajor
    @inlinable
    public var columnMajor: Self {
        // return self if already column major
        guard strides[Self.rank-1] < strides[Self.rank-2] else { return self }
        // compute column major strides for the last 2 dimensions
        var cmBounds = bounds
        cmBounds.swapAt(Self.rank-1, Self.rank-2)
        var cmStrides = cmBounds.sequentialStrides()
        cmStrides.swapAt(Self.rank-1, Self.rank-2)
        return Self(bounds, strides: cmStrides, storage: .colMajor)
    }
    
    //--------------------------------------------------------------------------
    /// repeated(repeatedBounds:
    @inlinable
    public func repeated(to repeatedBounds: Bounds) -> Self {
        // make sure the bounds are compatible
        assert({
            for i in 0..<Self.rank {
                if bounds[i] != 1 && bounds[i] != repeatedBounds[i] {
                    return false
                }
            }
            return true
        }(), "repeated tensor bounds must be either 1" +
            " or match the repeated tensor bounds")

        // compute strides, setting stride to 0 for repeated dimensions
        var repeatedStrides = Bounds.zero
        for i in 0..<Self.rank where repeatedBounds[i] == bounds[i] {
            repeatedStrides[i] = strides[i]
        }
        
        // it is sequential only for vectors
        return Self(repeatedBounds, strides: repeatedStrides,
                    storage: self.order)
    }

    //--------------------------------------------------------------------------
    /// transposed(with permutations:
    /// Returns a new data shape where the bounds and strides are permuted
    /// - Parameter permutations: the indice order mapping. `count` must
    ///   equal `rank`
    /// - Returns: transposed/permuted shape
    /// - Precondition: Each value in `permutations` must be in the range
    ///   `-rank..<rank`
    @inlinable
    public func transposed(with permutations: Bounds? = nil) -> Self {
        guard Self.rank > 1 else { return self }
        var newBounds = bounds
        var newStrides = strides

        // determine the new bounds and strides
        if let perm = permutations {
            let mapping = Self.makePositive(bounds: perm)
            for index in 0..<Self.rank {
                newBounds[index] = bounds[mapping[index]]
                newStrides[index] = strides[mapping[index]]
            }
        } else {
            // simple swap of last two dimensions
            newBounds.swapAt(Self.rank-1, Self.rank-2)
            newStrides.swapAt(Self.rank-1, Self.rank-2)
        }
        
        return Self(newBounds, strides: newStrides, storage: self.order)
    }
}

//==============================================================================
// Collection
extension ShapeProtocol
{
    @inlinable
    public var startIndex: ShapeIndex<Bounds> {
        ShapeIndex<Bounds>(Bounds.zero, 0)
    }

    @inlinable
    public var endIndex: ShapeIndex<Bounds> {
        ShapeIndex<Bounds>(Bounds.zero, count)
    }
    
    // returns the strided linear index corresponding
    // to the n-dimensional logical position
    @inlinable
    public subscript(index: ShapeIndex<Bounds>) -> Int {
        if isSequential {
            return index.sequenceIndex
        } else {
            return (index.position &* strides).wrappedSum()
        }
    }

    @inlinable
    public func index(after i: ShapeIndex<Bounds>) -> ShapeIndex<Bounds> {
        var next = i
        next.sequenceIndex += 1
        if !isSequential {
            next.position.increment(boundedBy: bounds)
        }
        return next
    }
}

//==============================================================================
// Equatable
extension ShapeProtocol {
    @inlinable
    public static func == (_ lhs: Self, _ rhs: [Int]) -> Bool {
        lhs.array == rhs
    }
}

//==============================================================================
/// ShapeIndex
public struct ShapeIndex<Bounds>: Comparable where Bounds: ShapeBounds {
    /// the logical position along each axis
    public var position: Bounds
    /// linear sequence position
    public var sequenceIndex: Int

    //------------------------------------
    // initializers
    @inlinable
    public init(_ position: Bounds, _ sequenceIndex: Int) {
        self.position = position
        self.sequenceIndex = sequenceIndex
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

//==============================================================================
// Shape
public struct Shape<Bounds: ShapeBounds>: ShapeProtocol
{
    public typealias Index = ShapeIndex<Bounds>
    
    // properties
    public let count: Int
    public let bounds: Bounds
    public let isSequential: Bool
    public let spanCount: Int
    public let order: StorageOrder
    public let strides: Bounds
    
    @inlinable
    public init(bounds: Bounds, strides: Bounds?, storage order: StorageOrder){
        assert(bounds.min() > 0, _messageInvalidBounds)
        self.bounds = bounds
        self.count = bounds.elementCount()
        self.order = order
        let sequentialStrides = bounds.sequentialStrides()

        if let callerStrides = strides {
            self.strides = callerStrides
            self.spanCount =  ((bounds &- 1) &* callerStrides).wrappedSum() + 1
            self.isSequential = callerStrides == sequentialStrides
        } else {
            self.strides = sequentialStrides
            self.spanCount = self.count
            self.isSequential = true
        }
    }
}
