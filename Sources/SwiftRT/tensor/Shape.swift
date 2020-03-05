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
// messages
@usableFromInline
let _messageInvalidBounds = "bounding dimensions must be greater than 0"

//==============================================================================
// ShapeBounds
public protocol ShapeBounds: SIMD {
    /// a tuple type used for rank invariant bounds initialization
    associatedtype Tuple
    /// the number of bounding dimensions
    static var rank: Int { get }
    
    init(_ s: Tuple)
}

//==============================================================================
// ShapeBounds extensions
extension ShapeBounds {
    /// instance member access
    @inlinable @_transparent
    var count: Int { Self.rank }

    /// helper
    @inlinable @_transparent
    mutating func swapAt(_ a: Int, _ b: Int) {
        let tmp = self[a]
        self[a] = self[b]
        self[b] = tmp
    }
    
    @inlinable @_transparent
    init?(_ s: Tuple?) {
        guard let s = s else { return nil }
        self.init(s)
    }
}

extension ShapeBounds where Scalar: FixedWidthInteger {
    //--------------------------------------------------------------------------
    /// `sequentialStrides`
    /// computes the row major sequential strides
    @inlinable
    func sequentialStrides() -> Self {
        var strides = Self.one
        for i in stride(from: Self.rank - 1, through: 1, by: -1) {
            strides[i - 1] = self[i] * strides[i]
        }
        return strides
    }
    
    //--------------------------------------------------------------------------
    /// `product`
    /// computes the product of the bounds scalars values
    @inlinable
    func product() -> Scalar {
        indices.reduce(into: 1) { $0 &*= self[$1] }
    }
}

//==============================================================================
// ShapeBounds SIMD extensions
extension SIMD1: ShapeBounds {
    public typealias Tuple = (Scalar)
    
    @inlinable @_transparent
    public static var rank: Int { 1 }

    @inlinable
    public init(_ s: Tuple) {
        self.init()
        self[0] = s
    }
}

extension SIMD2: ShapeBounds {
    public typealias Tuple = (Scalar, Scalar)

    @inlinable @_transparent
    public static var rank: Int { 2 }
    
    @inlinable
    public init(_ s: Tuple) {
        self.init()
        self[0] = s.0
        self[1] = s.1
    }
}

extension SIMD3: ShapeBounds {
    public typealias Tuple = (Scalar, Scalar, Scalar)

    @inlinable @_transparent
    public static var rank: Int { 3 }

    @inlinable
    public init(_ s: Tuple) {
        self.init()
        self[0] = s.0
        self[1] = s.1
        self[2] = s.2
    }
}

extension SIMD4: ShapeBounds {
    public typealias Tuple = (Scalar, Scalar, Scalar, Scalar)

    @inlinable @_transparent
    public static var rank: Int { 4 }

    @inlinable
    public init(_ s: Tuple) {
        self.init()
        self[0] = s.0
        self[1] = s.1
        self[2] = s.2
        self[3] = s.3
    }
}

extension SIMD5: ShapeBounds {
    public typealias Tuple = (Scalar, Scalar, Scalar, Scalar, Scalar)

    @inlinable @_transparent
    public static var rank: Int { 5 }

    @inlinable
    public init(_ s: Tuple) {
        self.init()
        self[0] = s.0
        self[1] = s.1
        self[2] = s.2
        self[3] = s.3
        self[4] = s.4
    }
}

//==============================================================================
/// ShapeProtocol
public protocol ShapeProtocol: Codable, Equatable, Collection
    where Element == Int
{
    // types
    associatedtype Bounds: ShapeBounds where Bounds.Scalar == Int

    //--------------------------------------------------------------------------
    // properties
    /// The bounds of the shape in each dimension
    var bounds: Bounds { get }
    /// the number of elements in the shape
    var count: Int { get }
    /// `true` if indexing is row sequential for performance
    var isSequential: Bool { get }
    /// The strided number of elements spanned by the shape
    var spanCount: Int { get }
    /// The distance to the next element for each dimension
    var strides: Bounds { get }
        
    //--------------------------------------------------------------------------
    /// Fully specified initializer
    /// - Parameter bounds: bounds of the shape in each dimension
    /// - Parameter strides: the distance to the next element in each dimension
    init(bounds: Bounds, strides: Bounds?)
    /// Expanding initializer
    /// - Parameter expanding: the lower order shape to expand
    /// - Parameter axes: the set of axes to be expanded
    init<S>(expanding other: S, alongAxes axes: Set<Int>?)
        where S: ShapeProtocol
    /// Flattening initializer
    /// - Parameter flattening: the higher order shape to flatten
    init<S>(flattening other: S) where S: ShapeProtocol
    /// Squeezing initializer
    /// - Parameter squeezing: the higher order shape to squeeze
    /// - Parameter axes: the set of axes to be squeezed
    init<S>(squeezing other: S, alongAxes axes: Set<Int>?)
        where S: ShapeProtocol
}

//==============================================================================
/// ShapeProtocol extensions
public extension ShapeProtocol {
    //--------------------------------------------------------------------------
    /// the static rank of the shape
    @inlinable
    @_transparent
    static var rank: Int { Bounds.rank }

    //--------------------------------------------------------------------------
    // computed properties
    /// array
    @inlinable var array: [Int] { [Int](self) }
    /// `true` if the shape has zero elements
    @inlinable
    var isEmpty: Bool { count == 0 }
    /// `true` if the shape has one element
    @inlinable
    var isScalar: Bool { count == 1 }
    /// the number of items in extent 0
    @inlinable
    var items: Int { bounds[0] }
    /// returns a dense version of self
    @inlinable
    var dense: Self { isSequential ? self : Self(bounds: bounds) }

    @inlinable
    static func == (_ lhs: Self, _ rhs: [Int]) -> Bool {
        lhs.array == rhs
    }
    
    //--------------------------------------------------------------------------
    // computeSpanCount
    // A sub view may cover a wider range of parent element indexes
    // than the number of dense elements defined by the bounds of the view
    // due to striding.
    // The span of the bounds is the linear index of the last index + 1
    @inlinable
    static func computeSpanCount(_ bounds: Bounds, _ strides: Bounds) -> Int {
        ((bounds &- 1) &* strides).wrappedSum() + 1
    }

    //--------------------------------------------------------------------------
    // linearIndex
    @inlinable
    func linearIndex(of position: Bounds) -> Int {
        (position &* strides).wrappedSum()
    }

    //--------------------------------------------------------------------------
    // init(bounds:
    @inlinable
    init(bounds: Bounds) {
        self.init(bounds: bounds, strides: nil)
    }
    
    //--------------------------------------------------------------------------
    // init with tuples
    @inlinable
    init(_ bounds: Bounds.Tuple) {
        self.init(bounds: Bounds(bounds), strides: nil)
    }

    @inlinable
    init?(_ bounds: Bounds.Tuple?) {
        guard let bounds = bounds else { return nil }
        self.init(bounds: Bounds(bounds), strides: nil)
    }

    //--------------------------------------------------------------------------
    // init(expanding:
    @inlinable
    init<S>(expanding other: S, alongAxes axes: Set<Int>? = nil)
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
        self.init(bounds: newBounds, strides: newStrides)
    }
    
    //--------------------------------------------------------------------------
    // init(indenting:
    @inlinable
    init<S>(indenting other: S) where S: ShapeProtocol
    {
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
        
        self.init(bounds: newBounds, strides: newStrides)
    }
    
    //--------------------------------------------------------------------------
    // init(padding:
    @inlinable
    init<S>(padding other: S) where S: ShapeProtocol {
        assert(S.rank < Self.rank, "can only pad lower ranked shapes")
        
        // Self and other are different ranks so we copy the leading elements
        var newBounds = Bounds.one
        var newStrides = Bounds.one
        for i in 0..<S.rank {
            newBounds[i] = other.bounds[i]
            newStrides[i] = other.strides[i]
        }
        self.init(bounds: newBounds, strides: newStrides)
    }
    
    //--------------------------------------------------------------------------
    // init(squeezing:
    @inlinable
    init<S>(squeezing other: S, alongAxes axes: Set<Int>? = nil)
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
        self.init(bounds: newBounds, strides: newStrides)
    }
    
    //--------------------------------------------------------------------------
    /// joined
    /// - Parameter others: array of data shapes to join
    /// - Parameter axis: the joining axis
    /// - Returns: returns a new shape that is the join with the others
    @inlinable
    func joined(with others: [Self], alongAxis axis: Int) -> Self {
        var newBounds = bounds
        newBounds[axis] += others.reduce(into: 0) { $0 += $1.bounds[axis] }
        return Self(bounds: newBounds)
    }
    
    //--------------------------------------------------------------------------
    /// makePositive(bounds:
    /// The user can specify indices from `-rank..<rank`.
    /// Negative numbers reference dimensions from the end of `bounds`
    /// This ensures they are resolved to positive values.
    @inlinable
    static func makePositive(bounds: Bounds) -> Bounds {
        var positive = bounds
        for i in 0..<Bounds.rank where positive[i] < 0 {
            positive[i] += Bounds.rank
        }
        return positive
    }

    //--------------------------------------------------------------------------
    /// contains
    @inlinable
    func contains(_ point: Bounds) -> Bool {
        linearIndex(of: point) <= spanCount
    }

    @inlinable
    func contains(other: Self) -> Bool {
        other.spanCount <= spanCount
    }

    //--------------------------------------------------------------------------
    /// columnMajor
    @inlinable
    var columnMajor: Self {
        // return self if already column major
        guard strides[Self.rank-1] < strides[Self.rank-2] else { return self }
        // compute column major strides for the last 2 dimensions
        var cmBounds = bounds
        cmBounds.swapAt(Self.rank-1, Self.rank-2)
        var cmStrides = cmBounds.sequentialStrides()
        cmStrides.swapAt(Self.rank-1, Self.rank-2)
        return Self(bounds: bounds, strides: cmStrides)
    }
    
    //--------------------------------------------------------------------------
    /// repeated(repeatedBounds:
    @inlinable
    func repeated(to repeatedBounds: Bounds) -> Self {
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
        return Self(bounds: repeatedBounds, strides: repeatedStrides)
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
    func transposed(with permutations: Bounds? = nil) -> Self {
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
        
        return Self(bounds: newBounds, strides: newStrides)
    }

}

//==============================================================================
// ShapeProtocol Collection extension
// *** Note: These seem to get inlined correctly, so they don't need to be
// on the struct
extension ShapeProtocol where Index == ShapeIndex<Bounds> {
    @inlinable
    public var startIndex: Index { Index(Bounds.zero, sequenceIndex: 0) }

    @inlinable
    public var endIndex: Index { Index(Bounds.zero, sequenceIndex: count) }
    
    // returns the strided linear index corresponding
    // to the n-dimensional logical position
    @inlinable
    public subscript(index: Index) -> Int {
        isSequential ? index.sequenceIndex :
            (index.position &* strides).wrappedSum()
    }
}

//==============================================================================
/// ShapeIndex
public struct ShapeIndex<Bounds>: Comparable
    where Bounds: ShapeBounds
{
    //------------------------------------
    /// the logical position along each axis
    public var position: Bounds
    /// linear sequence position
    public var sequenceIndex: Int

    //------------------------------------
    // initializers
    @inlinable
    public init(_ position: Bounds, sequenceIndex: Int) {
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
// Shape1
public struct Shape1: ShapeProtocol {
    public typealias Bounds = SIMD1<Int>
    public typealias Index = Int
    
    // properties
    public let count: Int
    public let bounds: Bounds
    public let isSequential: Bool
    public let spanCount: Int
    public let strides: Bounds

    @inlinable
    public init(bounds: Bounds, strides: Bounds? = nil) {
        assert(bounds.min() > 0, _messageInvalidBounds)
        self.bounds = bounds
        self.count = bounds[0]

        if let callerStrides = strides {
            self.strides = callerStrides
            self.spanCount = ((bounds[0] - 1) * callerStrides[0]) + 1
        } else {
            self.strides = Bounds.one
            self.spanCount = self.count
        }
        self.isSequential = self.strides[0] == 1
    }

    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isSequential, "cannot flatten non sequential data")
        self.init(bounds: Bounds(other.count))
    }
    
    //--------------------------------------------------------------------------
    // indexing
    @inlinable public var startIndex: Index { 0 }

    @inlinable public var endIndex: Index { count }
    
    @inlinable public func index(after i: Index) -> Index { i + 1 }

    // TODO: look into the idea of Sequential Shapes to eliminate
    // unneeded stride multiplication, it costs 30% indexing overhead
    @inlinable public subscript(index: Index) -> Int { index * strides[0] }
}

//==============================================================================
// Shape2
public struct Shape2: ShapeProtocol {
    public typealias Bounds = SIMD2<Int>
    public typealias Index = ShapeIndex<Bounds>
    
    // properties
    public let count: Int
    public let bounds: Bounds
    public let isSequential: Bool
    public let spanCount: Int
    public let strides: Bounds

    @inlinable
    public init(bounds: Bounds, strides: Bounds? = nil) {
        assert(bounds.min() > 0, _messageInvalidBounds)
        // ******** original **********
//        self.bounds = bounds
//        self.count = bounds.product()
//        self.isSequential = isSequential
//        self.strides = strides ?? bounds.sequentialStrides()
//        self.spanCount = Self.computeSpanCount(self.bounds, self.strides)

        // 95% Faster!!!
        self.bounds = bounds
        self.count = bounds[0] * bounds[1]
        let sequentialStrides = Bounds(bounds[1], 1)
        assert(sequentialStrides == bounds.sequentialStrides())

        if let callerStrides = strides {
            self.strides = callerStrides
            self.spanCount = ((bounds &- 1) &* callerStrides).wrappedSum() + 1
        } else {
            self.strides = sequentialStrides
            self.spanCount = self.count
        }
        self.isSequential = self.strides == sequentialStrides
    }

    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isSequential, "cannot flatten non sequential data")
        assert(S.rank >= Self.rank, "cannot flatten bounds of lower rank")
        self.init(bounds: Bounds(
            other.bounds[0],
            other.count / other.bounds[0]
        ))
    }
    
    //--------------------------------------------------------------------------
    // index(i:
    // Note: this does not get inlined unless part of the struct
    @inlinable
    public func index(after i: Index) -> Index {
        var position = i.position

        if !isSequential {
            // a recursive algorithm was ~55x slower
            position[1] += 1
            
            if position[1] == bounds[1] {
                position[1] = 0
                position[0] += 1
            }
        }
        return Index(position, sequenceIndex: i.sequenceIndex + 1)
    }
}

//==============================================================================
// Shape3
public struct Shape3: ShapeProtocol {
    public typealias Bounds = SIMD3<Int>
    public typealias Index = ShapeIndex<Bounds>
    
    // properties
    public let count: Int
    public let bounds: Bounds
    public let isSequential: Bool
    public let spanCount: Int
    public let strides: Bounds

    @inlinable
    public init(bounds: Bounds, strides: Bounds? = nil) {
        assert(bounds.min() > 0, _messageInvalidBounds)
        self.bounds = bounds
        self.count = bounds[0] * bounds[1] * bounds[2]
        let sequentialStrides = Bounds(bounds[1] * bounds[2], bounds[2], 1)
        assert(sequentialStrides == bounds.sequentialStrides())

        if let callerStrides = strides {
            self.strides = callerStrides
            self.spanCount = ((bounds &- 1) &* callerStrides).wrappedSum() + 1
        } else {
            self.strides = sequentialStrides
            self.spanCount = self.count
        }
        self.isSequential = self.strides == sequentialStrides
    }

    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isSequential, "cannot flatten non sequential data")
        assert(S.rank >= Self.rank, "cannot flatten bounds of lower rank")
        
        self.init(bounds: Bounds(
            other.bounds[0],
            other.bounds[1],
            (2..<S.rank).reduce(into: 0) { $0 += other.bounds[$1] }
        ))
    }
    
    //--------------------------------------------------------------------------
    // index(i:
    // Note: this does not get inlined unless part of the struct
    @inlinable
    public func index(after i: Index) -> Index {
        var position = i.position
        
        if !isSequential {
            // a recursive algorithm was ~55x slower
            position[2] += 1
            if position[2] == bounds[2] {
                position[2] = 0
                position[1] += 1
                
                if position[1] == bounds[1] {
                    position[1] = 0
                    position[0] += 1
                }
            }
        }
        
        return Index(position, sequenceIndex: i.sequenceIndex + 1)
    }
}

//==============================================================================
// Shape4
public struct Shape4: ShapeProtocol {
    public typealias Bounds = SIMD4<Int>
    public typealias Index = ShapeIndex<Bounds>
    
    // properties
    public let count: Int
    public let bounds: Bounds
    public let isSequential: Bool
    public let spanCount: Int
    public let strides: Bounds

    @inlinable
    public init(bounds: Bounds, strides: Bounds? = nil) {
        assert(bounds.min() > 0, _messageInvalidBounds)
        self.bounds = bounds
        self.count = bounds[0] * bounds[1] * bounds[2] * bounds[3]
        let sequentialStrides = Bounds(
            bounds[1] * bounds[2] * bounds[3],
            bounds[2] * bounds[3],
            bounds[3],
            1)
        assert(sequentialStrides == bounds.sequentialStrides())

        if let callerStrides = strides {
            self.strides = callerStrides
            self.spanCount = ((bounds &- 1) &* callerStrides).wrappedSum() + 1
        } else {
            self.strides = sequentialStrides
            self.spanCount = self.count
        }
        self.isSequential = self.strides == sequentialStrides
    }

    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isSequential, "cannot flatten non sequential data")
        assert(S.rank >= Self.rank, "cannot flatten bounds of lower rank")
        
        self.init(bounds: Bounds(
            other.bounds[0],
            other.bounds[1],
            other.bounds[2],
            (3..<S.rank).reduce(into: 0) { $0 += other.bounds[$1] }
        ))
    }
    
    //--------------------------------------------------------------------------
    // index(i:
    // Note: this does not get inlined unless part of the struct
    @inlinable
    public func index(after i: Index) -> Index {
        var position = i.position
        
        if !isSequential {
            // a recursive algorithm was ~55x slower
            position[3] += 1
            if position[3] == bounds[3] {
                position[3] = 0
                position[2] += 1
                
                if position[2] == bounds[2] {
                    position[2] = 0
                    position[1] += 1
                    
                    if position[1] == bounds[1] {
                        position[1] = 0
                        position[0] += 1
                    }
                }
            }
        }
        
        return Index(position, sequenceIndex: i.sequenceIndex + 1)
    }
}

//==============================================================================
// Shape5
public struct Shape5: ShapeProtocol {
    public typealias Bounds = SIMD5<Int>
    public typealias Index = ShapeIndex<Bounds>
    
    // properties
    public let count: Int
    public let bounds: Bounds
    public let isSequential: Bool
    public let spanCount: Int
    public let strides: Bounds

    @inlinable
    public init(bounds: Bounds, strides: Bounds? = nil) {
        assert(bounds.min() > 0, _messageInvalidBounds)
        self.bounds = bounds
        // TODO: verify if unrolling is actually faster, or using partial
        // results, or if the compiler figures it out and does the right thing
        self.count = bounds[0] * bounds[1] * bounds[2] * bounds[3] * bounds[4]
        let sequentialStrides = Bounds((
            bounds[1] * bounds[2] * bounds[3] * bounds[4],
            bounds[2] * bounds[3] * bounds[4],
            bounds[3] * bounds[4],
            bounds[4],
            1))
        assert(sequentialStrides == bounds.sequentialStrides())

        if let callerStrides = strides {
            self.strides = callerStrides
            self.spanCount = ((bounds &- 1) &* callerStrides).wrappedSum() + 1
        } else {
            self.strides = sequentialStrides
            self.spanCount = self.count
        }
        self.isSequential = self.strides == sequentialStrides
    }

    //--------------------------------------------------------------------------
    // init(flattening:
    @inlinable
    public init<S>(flattening other: S) where S: ShapeProtocol {
        assert(other.isSequential, "cannot flatten non sequential data")
        assert(S.rank >= Self.rank, "cannot flatten bounds of lower rank")
        
        self.init(bounds: Bounds((
            other.bounds[0],
            other.bounds[1],
            other.bounds[2],
            other.bounds[3],
            (4..<S.rank).reduce(into: 0) { $0 += other.bounds[$1] }
        )))
    }
    
    //--------------------------------------------------------------------------
    // index(i:
    // Note: this does not get inlined unless part of the struct, and using
    // a recursive algorithm is ~55x slower
    @inlinable
    public func index(after i: Index) -> Index {
        var position = i.position
        
        if !isSequential {
            // a recursive algorithm was ~55x slower
            position[4] += 1
            if position[4] == bounds[4] {
                position[4] = 0
                position[3] += 1
                
                if position[3] == bounds[3] {
                    position[3] = 0
                    position[2] += 1
                    
                    if position[2] == bounds[2] {
                        position[2] = 0
                        position[1] += 1
                        
                        if position[1] == bounds[1] {
                            position[1] = 0
                            position[0] += 1
                        }
                    }
                }
            }
        }
        return Index(position, sequenceIndex: i.sequenceIndex + 1)
    }
}
