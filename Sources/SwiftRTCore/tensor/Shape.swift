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
// Shape
public protocol TensorShape: SIMD where Scalar == Int {
  /// simd storage type that lowers to llvm simd types
  associatedtype Storage: SIMDStorage where Storage.Scalar == Int
  /// a ranked tuple convenience type used for api parameters
  associatedtype Tuple

  //---------------------------------
  // properties
  /// the number of bounding dimensions for the shape
  static var rank: Int { get }
  /// a tuple of zeros
  static var zeroTuple: Tuple { get }
  /// a tuple of ones
  static var oneTuple: Tuple { get }

  /// conversion to Int32 to support drivers
  var _storage: Storage { get set }

  //---------------------------------
  // initializers
  /// Creates a shape with zero in all lanes.
  init()
  /// Creates a shape described by a rank specific tuple value
  init(_ shape: Tuple)
  /// Optionally creates a shape described by a rank specific tuple value
  init?(_ shape: Tuple?)
}

//==============================================================================
// messages
@usableFromInline let _messageInvalidShape = "shape dimensions must be greater than 0"

//==============================================================================
// Shape comparative
extension TensorShape where Self: SIMD, Scalar == Int {
  @inlinable public static func <(_ lhs: Self, _ rhs: Self) -> Bool {
    lhs.indices.reduce(into: true) { $0 = $0 && (lhs[$1] < rhs[$1]) }
  }
  
  @inlinable public static func <=(_ lhs: Self, _ rhs: Self) -> Bool {
    lhs.indices.reduce(into: true) { $0 = $0 && (lhs[$1] <= rhs[$1]) }
  }

  @inlinable public static func >(_ lhs: Self, _ rhs: Self) -> Bool {
    lhs.indices.reduce(into: true) { $0 = $0 && (lhs[$1] > rhs[$1]) }
  }
  
  @inlinable public static func >=(_ lhs: Self, _ rhs: Self) -> Bool {
    lhs.indices.reduce(into: true) { $0 = $0 && (lhs[$1] >= rhs[$1]) }
  }
}

//==============================================================================
// extensions
extension TensorShape {
  //--------------------------------------------------------------------------
  // Counts
  
  /// the number of scalars in the simd vector
  @_transparent public static var scalarCount: Int { Self.rank }
  @_transparent public var scalarCount: Int { Self.rank }
  
  /// the index of the last value in the simd vector
  @_transparent public static var lastIndex: Int { Self.rank - 1 }

  /// elementCount
  /// - Returns: the number of spatial elements bounded by the shape
  @inlinable public func elementCount() -> Int {
    self.reduce(into: 1, &*=)
  }

  //--------------------------------------------------------------------------
  // initializers
  @inlinable public init(integerLiteral value: Scalar) {
    self.init(repeating: value)
  }

  /// init(
  /// initiailze with an optional tuple shape
  @inlinable public init?(_ shape: Tuple?) {
    guard let shape = shape else { return nil }
    self.init(shape)
  }
    
  /// init(flattening:
  /// - Parameters:
  ///  - other: the shape to flatten
  @inlinable public init<S: TensorShape>(flattening other: S) {
    assert(Self.rank < S.rank, "cannot flatten shape of lower rank")
    
    // copy other's leading dimensions
    self = Self.zero
    for i in indices {
      self[i] = other[i]
    }
    
    // get product of the remaining dimensions
    for j in Self.rank..<S.rank {
      self[Self.lastIndex] &*= other[j]
    }
  }

  //--------------------------------------------------------------------------
  // interchange with Swift arrays
  @inlinable public init(_ shape: [Int32]) {
    assert(shape.count == Self.rank, "rank mismatch")
    self.init()
    indices.forEach { self[$0] = Int(shape[$0]) }
  }
  
  // TODO: should go away
  @inlinable public var asInt32: [Int32] {
    var values = [Int32]()
    indices.forEach { values.append(Int32(self[$0])) }
    return values
  }

  /// copy to Swift Array
  @inlinable public var array: [Scalar] {
    var a = [Scalar]()
    indices.forEach { a.append(self[$0]) }
    return a
  }

  //--------------------------------------------------------------------------
  // Indexing
  
  /// Accesses the scalar at the specified position.
  public subscript(index: Int) -> Scalar {
    @_transparent get {
      precondition(indices.contains(index), "Shape\(Self.rank) index is out of bounds: \(index)")
      return _storage[index]
    }
    @_transparent set {
      precondition(indices.contains(index), "Shape\(Self.rank) index is out of bounds: \(index)")
      _storage[index] = newValue
    }
  }

  /// incremented(between lower:and upper:
  /// generic n-dimensional position increment function when the vector
  /// is being used as an index
  /// - Parameters:
  ///  - lower: the lower bound for the space
  ///  - upper: the upper bound for the space
  /// - Returns: the next logical position within the nD space
  ///
  @inlinable public func incremented(
    between lower: Self,
    and upper: Self
  ) -> Self {
    assert(self >= lower && self < upper, "index must be between lower and upper bounds")
    var next = self
    for i in indices.reversed() {
      next[i] &+= 1
      if next[i] == upper[i] {
        next[i] = lower[i]
      } else {
        // early exit
        return next
      }
    }

    // if the while loop doesn't early return then this is the end position
    next[0] &+= 1
    return next
  }
  
  //--------------------------------------------------------------------------
  /// transpose last two dimensions
  @inlinable public var t: Self {
    var transposed = self
    transposed.swapAt(Self.lastIndex, Self.lastIndex - 1)
    return transposed
  }
  
  //--------------------------------------------------------------------------
  /// swapAt(_:_
  @inlinable public mutating func swapAt(_ a: Int, _ b: Int) {
    let tmp = self[a]
    self[a] = self[b]
    self[b] = tmp
  }

  //--------------------------------------------------------------------------
  /// `reduce
  /// - Parameters:
  ///  - initialResult: the initial result value
  ///  - updateAccumulatingResult: accumulation functions
  /// - Returns: the reduced simd vector scalars
  @inlinable public func reduce(
    into initialResult: Scalar,
    _ updateAccumulatingResult: (inout Scalar, Scalar) -> Void
  ) -> Scalar {
    indices.reduce(into: initialResult) {
      updateAccumulatingResult(&$0, self[$1])
    }
  }

  //--------------------------------------------------------------------------
  /// index
  /// computes a linear index by multiplying `self` by `strides`
  /// - Parameters:
  ///  - strides: the strides for shape
  /// - Returns: linear strided index
  @inlinable public func index(stridedBy strides: Self) -> Int {
    (self &* strides).wrappedSum()
  }

  //--------------------------------------------------------------------------
  /// spanCount
  /// computes the number of physical buffer elements spanned when `self` is
  /// a shape multiplied by strides.
  /// - Parameters:
  ///  - strides: the strides for shape
  /// - Returns: the distance from the first element's linear storage index
  ///   to the last
  @inlinable public func spanCount(stridedBy strides: Self) -> Int {
    ((self &- 1) &* strides).wrappedSum() &+ 1
  }

  //--------------------------------------------------------------------------
  /// `strides(order:`
  /// computes the strides needed to index the specified storage order
  // TODO: rethink this because it only makes sense for row and col orders
  @inlinable public func strides(for order: Order) -> Self {
    guard Self.rank > 1 else { return Self.one }

    func computeStrides(for shape: Self) -> Self {
      // just use shape to reserve some storage space for strides
      var strides = shape
      var dim = Self.lastIndex
      var shapeStride = 1
      while dim >= 0 {
        strides[dim] = shapeStride
        shapeStride &*= shape[dim]
        dim &-= 1
      }
      return strides
    }

    switch order {
    case .row: return computeStrides(for: self)
    case .col:
      var shape = self
      shape.swapAt(Self.lastIndex, Self.lastIndex - 1)
      var strides = computeStrides(for: shape)
      strides.swapAt(Self.lastIndex, Self.lastIndex - 1)
      return strides
    default: fatalError("not implemented yet")
    }
  }

  @inlinable public func strides() -> Self {
    strides(for: .row)
  }

  //--------------------------------------------------------------------------
  /// `areSequential`
  /// - Parameter shape: the bounding shape for the strides
  /// - Returns: `true` if `self` is the sequential strides for the given shape
  @inlinable public func areSequential(for shape: Self) -> Bool {
    var dim = Self.lastIndex
    var shapeStride = 1
    while dim >= 0 {
      if self[dim] != shapeStride && shape[dim] > 1 {
        return false
      }
      shapeStride &*= shape[dim]
      dim &-= 1
    }
    return true
  }

  //--------------------------------------------------------------------------
  /// withUnsafePointer(_:
  /// - Returns: a pointer to the simd vector scalars. Primarily used
  /// to pass a pointer to a C driver
  @inlinable public func withUnsafePointer<Result>(
    _ body: (UnsafePointer<Scalar>) -> Result
  ) -> Result {
    Swift.withUnsafePointer(to: _storage) {
      body(UnsafeRawPointer($0).assumingMemoryBound(to: Scalar.self))
    }
  }
  
  //--------------------------------------------------------------------------
  public var description: String {
    var desc = "Shape\(Self.rank)("
    for i in 0..<Self.lastIndex { desc += "\(self[i]), " }
    desc += "\(self[Self.lastIndex]))"
    return desc
  }
}

//==============================================================================
/// Shape1
/// Represents the shape of a 1D element space
public struct Shape1: TensorShape, ExpressibleByIntegerLiteral {
  // types
  public typealias Tuple = (Int)

  // properties
  public var _storage: Int.SIMD2Storage
  public typealias MaskStorage = SIMD2<Int.SIMDMaskScalar>
  @_transparent public static var zeroTuple: Tuple { (0) }
  @_transparent public static var oneTuple: Tuple { (1) }
  @_transparent public static var rank: Int { 1 }
  
  //------------------------------------------
  // initializers
  /// Creates a vector with zero in all lanes.
  @_transparent public init() { _storage = Scalar.SIMD2Storage() }

  @_transparent public init(_ v0: Int) {
    self.init()
    self[0] = v0
  }
}

//==============================================================================
/// Shape2
/// Represents the shape of a 2D element space
public struct Shape2: TensorShape, ExpressibleByIntegerLiteral {
  // types
  public typealias Tuple = (Int, Int)
  public typealias M1 = Shape1

  // properties
  public var _storage: Int.SIMD2Storage
  public typealias MaskStorage = SIMD2<Int.SIMDMaskScalar>
  @_transparent public static var zeroTuple: Tuple { (0, 0) }
  @_transparent public static var oneTuple: Tuple { (1, 1) }
  @_transparent public static var rank: Int { 2 }

  //------------------------------------------
  // initializers
  /// Creates a vector with zero in all lanes.
  @_transparent public init() { _storage = Scalar.SIMD2Storage() }
  
  @_transparent public init(_ v0: Int, _ v1: Int) {
    self.init()
    self[0] = v0
    self[1] = v1
  }

  @_transparent public init(_ shape: Tuple) {
    self.init(shape.0, shape.1)
  }
}

//==============================================================================
/// Shape3
/// Represents the shape of a 3D element space
public struct Shape3: TensorShape, ExpressibleByIntegerLiteral {
  // types
  public typealias Tuple = (Int, Int, Int)
  public typealias M1 = Shape2
  
  // properties
  public var _storage: Int.SIMD4Storage
  public typealias MaskStorage = SIMD4<Int.SIMDMaskScalar>
  @_transparent public static var zeroTuple: Tuple { (0, 0, 0) }
  @_transparent public static var oneTuple: Tuple { (1, 1, 1) }
  @_transparent public static var rank: Int { 3 }
  
  //------------------------------------------
  // initializers
  /// Creates a vector with zero in all lanes.
  @_transparent public init() { _storage = Scalar.SIMD4Storage() }
  
  @_transparent public init(_ v0: Int, _ v1: Int, _ v2: Int) {
    self.init()
    self[0] = v0
    self[1] = v1
    self[2] = v2
  }
  
  @_transparent public init(_ shape: Tuple) {
    self.init(shape.0, shape.1, shape.2)
  }
}

//==============================================================================
/// Shape4
/// Represents the shape of a 4D element space
public struct Shape4: TensorShape, ExpressibleByIntegerLiteral {
  // types
  public typealias Tuple = (Int, Int, Int, Int)
  public typealias M1 = Shape3
  
  // properties
  public var _storage: Int.SIMD4Storage
  public typealias MaskStorage = SIMD4<Int.SIMDMaskScalar>
  @_transparent public static var zeroTuple: Tuple { (0, 0, 0, 0) }
  @_transparent public static var oneTuple: Tuple { (1, 1, 1, 1) }
  @_transparent public static var rank: Int { 4 }
  
  //------------------------------------------
  // initializers
  /// Creates a vector with zero in all lanes.
  @_transparent public init() { _storage = Scalar.SIMD4Storage() }
  
  @_transparent public init(_ v0: Int, _ v1: Int, _ v2: Int, _ v3: Int) {
    self.init()
    self[0] = v0
    self[1] = v1
    self[2] = v2
    self[3] = v3
  }
  
  @_transparent public init(_ shape: Tuple) {
    self.init(shape.0, shape.1, shape.2, shape.3)
  }
}

//==============================================================================
/// Shape5
/// Represents the shape of a 5D element space
public struct Shape5: TensorShape, ExpressibleByIntegerLiteral {
  // types
  public typealias Tuple = (Int, Int, Int, Int, Int)
  public typealias M1 = Shape4
  
  // properties
  public var _storage: Int.SIMD8Storage
  public typealias MaskStorage = SIMD8<Int.SIMDMaskScalar>
  @_transparent public static var zeroTuple: Tuple { (0, 0, 0, 0, 0) }
  @_transparent public static var oneTuple: Tuple { (1, 1, 1, 1, 1) }
  @_transparent public static var rank: Int { 5 }
  
  //------------------------------------------
  // initializers
  /// Creates a vector with zero in all lanes.
  @_transparent public init() { _storage = Scalar.SIMD8Storage() }
  
  @_transparent public init(_ v0: Int, _ v1: Int, _ v2: Int, _ v3: Int, _ v4: Int) {
    self.init()
    self[0] = v0
    self[1] = v1
    self[2] = v2
    self[3] = v3
    self[4] = v4
  }
  
  @_transparent public init(_ shape: Tuple) {
    self.init(shape.0, shape.1, shape.2, shape.3, shape.4)
  }
}

//==============================================================================
/// Shape6
/// Represents the shape of a 6D element space
public struct Shape6: TensorShape, ExpressibleByIntegerLiteral {
  // types
  public typealias Tuple = (Int, Int, Int, Int, Int, Int)
  public typealias M1 = Shape5
  
  // properties
  public var _storage: Int.SIMD8Storage
  public typealias MaskStorage = SIMD8<Int.SIMDMaskScalar>
  @_transparent public static var zeroTuple: Tuple { (0, 0, 0, 0, 0, 0) }
  @_transparent public static var oneTuple: Tuple { (1, 1, 1, 1, 1, 1) }
  @_transparent public static var rank: Int { 6 }
  
  //------------------------------------------
  // initializers
  /// Creates a vector with zero in all lanes.
  @_transparent public init() { _storage = Scalar.SIMD8Storage() }
  
  @_transparent public init(_ v0: Int, _ v1: Int, _ v2: Int, _ v3: Int, _ v4: Int, _ v5: Int) {
    self.init()
    self[0] = v0
    self[1] = v1
    self[2] = v2
    self[3] = v3
    self[4] = v4
    self[5] = v5
  }
  
  @_transparent public init(_ shape: Tuple) {
    self.init(shape.0, shape.1, shape.2, shape.3, shape.4, shape.5)
  }
}

