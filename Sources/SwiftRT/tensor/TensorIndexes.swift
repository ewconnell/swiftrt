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
// shaped positions and extents used for indexing and selection
public typealias NDPosition = [Int]
public typealias VectorPosition = Int
public typealias MatrixPosition = (r: Int, c: Int)
public typealias VolumePosition = (d: Int, r: Int, c: Int)
public typealias NCHWPosition = (i: Int, ch: Int, r: Int, c: Int)
public typealias NHWCPosition = (i: Int, r: Int, c: Int, ch: Int)

//==============================================================================
/// VectorIndex
public struct VectorIndex: TensorIndexing {
    // properties
    public var viewIndex: Int
    public var dataIndex: Int
    public let stride: Int
    
    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init<T>(view: T, at position: VectorPosition) where T: TensorView {
        viewIndex = position
        dataIndex = 0
        stride = Int(view.strides[0])
        computeDataIndex()
    }
    
    @inlinable
    public init<T>(endOf view: T) where T: TensorView {
        viewIndex = view.count
        stride = Int(view.strides[0])
        dataIndex = 0
        computeDataIndex()
    }
    
    //--------------------------------------------------------------------------
    /// computeDataIndex
    @inlinable
    public mutating func computeDataIndex() {
        dataIndex = viewIndex * stride
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable
    public func increment() -> VectorIndex {
        var next = self
        next.viewIndex += 1
        next.dataIndex += stride
        return next
    }
    
    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable
    public func advanced(by n: Int) -> VectorIndex {
        guard n != 1 else { return increment() }
        var next = self
        next.viewIndex += n
        next.computeDataIndex()
        return next
    }
}

//==============================================================================
/// MatrixIndex
public struct MatrixIndex: TensorIndexing {
    // properties
    public var viewIndex: Int
    public var dataIndex: Int
    public let rowExtent: Int
    public let rowStride: Int
    public let colExtent: Int
    public let colStride: Int
    public var row: Int
    public var col: Int
    
    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init<T>(view: T, at position: MatrixPosition) where T: TensorView {
        rowExtent = Int(view.extents[0])
        rowStride = Int(view.strides[0])
        row = position.r
        colExtent = Int(view.extents[1])
        colStride = Int(view.strides[1])
        col = position.c
        viewIndex = row * rowExtent + col * colExtent
        dataIndex = 0
        computeDataIndex()
    }

    @inlinable
    public init<T>(endOf view: T) where T: TensorView {
        rowExtent = Int(view.extents[0])
        rowStride = Int(view.strides[0])
        row = rowExtent
        colExtent = Int(view.extents[1])
        colStride = Int(view.strides[1])
        col = 0
        viewIndex = view.count
        dataIndex = 0
        computeDataIndex()
    }

    //--------------------------------------------------------------------------
    /// computeDataIndex
    @inlinable
    public mutating func computeDataIndex() {
        dataIndex = row * rowStride + col * colStride
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable
    public func increment() -> MatrixIndex {
        var next = self
        next.viewIndex += 1
        next.col += 1
        if next.col == colExtent {
            next.col = 0
            next.row += 1
            next.computeDataIndex()
        } else {
            next.dataIndex += colStride
        }
        return next
    }

    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable
    public func advanced(by n: Int) -> MatrixIndex {
        guard n != 1 else { return increment() }
        let jump = n.quotientAndRemainder(dividingBy: rowExtent)
        var next = self
        next.viewIndex += n
        next.row += jump.quotient
        next.col += jump.remainder
        next.computeDataIndex()
        return next
    }
}

//==============================================================================
/// VolumeIndex
public struct VolumeIndex: TensorIndexing {
    // properties
    public var dataIndex: Int
    public var viewIndex: Int
    public let depExtent: Int
    public let depStride: Int
    public let rowExtent: Int
    public let rowStride: Int
    public let colExtent: Int
    public let colStride: Int
    public var dep: Int
    public var row: Int
    public var col: Int

    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init<T>(view: T, at position: VolumePosition) where T: TensorView {
        depExtent = Int(view.extents[0])
        depStride = Int(view.strides[0])
        dep = position.d
        rowExtent = Int(view.extents[1])
        rowStride = Int(view.strides[1])
        row = position.r
        colExtent = Int(view.extents[2])
        colStride = Int(view.strides[2])
        col = position.c
        viewIndex = dep * depExtent + row * rowExtent + col * colExtent
        dataIndex = 0
        computeDataIndex()
    }
    
    @inlinable
    public init<T>(endOf view: T) where T: TensorView {
        depExtent = Int(view.extents[0])
        depStride = Int(view.strides[0])
        dep = Int(view.extents[0])
        rowExtent = Int(view.extents[1])
        rowStride = Int(view.strides[1])
        row = 0
        colExtent = Int(view.extents[2])
        colStride = Int(view.strides[2])
        col = 0
        viewIndex = view.count
        dataIndex = 0
        computeDataIndex()
    }
    
    //--------------------------------------------------------------------------
    /// computeDataIndex
    @inlinable
    public mutating func computeDataIndex() {
        dataIndex = dep * depStride + row * rowStride + col * colStride
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable
    public func increment() -> VolumeIndex {
        var next = self
        next.viewIndex += 1
        next.col += 1
        if next.col == colExtent {
            next.col = 0
            next.row += 1
            if next.row == rowExtent {
                next.row = 0
                next.dep += 1
            }
            next.computeDataIndex()
        } else {
            next.dataIndex += colStride
        }
        return next
    }
    
    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable
    public func advanced(by n: Int) -> VolumeIndex {
        guard n != 1 else { return increment() }
        var next = self
        next.viewIndex += n

        // update the depth, row, and column positions
        var jump = n.quotientAndRemainder(dividingBy: depExtent)
        let quotient = jump.quotient
        next.dep += quotient
        
        jump = quotient.quotientAndRemainder(dividingBy: rowExtent)
        next.row += jump.quotient
        next.col += jump.remainder
        
        // now set the index
        next.computeDataIndex()
        return next
    }
}

//==============================================================================
/// NDIndex
public struct NDIndex: TensorIndexing {
    // properties
    public var viewIndex: Int
    public var dataIndex: Int
    public var position: NDPosition
    public let extents: [Int]
    public let strides: [Int]
    
    //--------------------------------------------------------------------------
    // initializers
    @inlinable
    public init<T>(view: T, at position: NDPosition) where T: TensorView {
        assert(position.count == 1 || position.count == view.rank)
        extents = view.extents.array
        strides = view.strides.array
        self.position = position.count == 1 ? T.Shape.zeros.array : position
        viewIndex = zip(position, extents).reduce(0) { $0 + $1.0 * $1.1 }
        dataIndex = 0
        computeDataIndex()
    }
    
    @inlinable
    public init<T>(endOf view: T) where T: TensorView {
        position = [Int](repeating: 0, count: view.rank)
        position[0] = Int(view.extents[0])
        extents = view.extents.array
        strides = view.strides.array
        viewIndex = view.count
        dataIndex = 0
        computeDataIndex()
    }
    
    //--------------------------------------------------------------------------
    /// computeDataIndex
    @inlinable
    public mutating func computeDataIndex() {
        dataIndex = zip(position, strides).reduce(0) {
            $0 + $1.0 * $1.1
        }
    }
    
    //--------------------------------------------------------------------------
    /// increment
    /// incremental update of indexes used for iteration
    @inlinable
    public func increment() -> NDIndex {
        var next = self
        next.viewIndex += 1
        
        // increment the last dimension,
        // recursively working towards lower rank dimensions
        func nextPosition(for dim: Int) {
            next.position[dim] += 1
            if next.position[dim] == extents[dim] && dim > 0 {
                next.position[dim] = 0
                nextPosition(for: dim - 1)
            }
        }
        nextPosition(for: extents.count - 1)
        
        // TODO: this should be reconsidered for
        // the simple incremental case instead of full recompute
        next.computeDataIndex()
        return next
    }
    
    //--------------------------------------------------------------------------
    /// advanced(by n:
    /// bidirectional jump or movement
    @inlinable
    public func advanced(by n: Int) -> NDIndex {
        guard n != 1 else { return increment() }
        var next = self
        var distance = n
        next.viewIndex += n

        var jump: (quotient: Int, remainder: Int)
        for dim in (1..<extents.count).reversed() {
            jump = distance.quotientAndRemainder(dividingBy: extents[dim])
            next.position[dim] += jump.quotient
            distance = jump.remainder
            if dim == 1 {
                next.position[0] += jump.remainder
            }
        }
        next.computeDataIndex()
        return next
    }
}
