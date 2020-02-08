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

public extension TensorView where Self: VectorView {
    //--------------------------------------------------------------------------
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(index: Int) -> Element {
        get {
            view(at: makePositive(index: (index)),
                 extents: Shape.ones, strides: Shape.ones).element
        }
        set {
            // make sure a possibly repeated view is dense before getting
            // the extents and strides used for writing.
            makeDense(view: self)
            var view = mutableView(at: makePositive(index: (index)),
                                   extents: Shape.ones, strides: Shape.ones)
            view.element = newValue
        }
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int
        {
        get {
            let r = range.relativeTo(0..<extents[0])
            return self[(r.start), (r.end), (r.step)]
        }
        set {
            let r = range.relativeTo(0..<extents[0])
            self[(r.start), (r.end), (r.step)] = newValue
        }
    }
}

public extension TensorView {
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(range: UnboundedRange) -> Self { self }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get {
            let (start, end, steps) =
                getItemRange(range.relativeTo(0..<extents[0]))
            return self[start, end, steps]
        }
        set {
            let (start, end, steps) =
                getItemRange(range.relativeTo(0..<extents[0]))
            self[start, end, steps] = newValue
        }
    }
    
    @usableFromInline
    @_semantics("autodiff.nonvarying")
    internal func getItemRange(_ range: StridedRange<Int>) ->
        (Shape.Array, Shape.Array, Shape.Array)
    {
        var start = Shape.zeros
        var end = self.extents
        var steps = Shape.ones
        start[0] = range.start
        end[0] = range.end
        steps[0] = range.step
        return (start, end, steps)
    }
}

//==============================================================================
public extension TensorView {
    //--------------------------------------------------------------------------
    /// getExtents(from:to:
    /// computes the extents and strides from the specified bounds and steps
    /// - Parameter lower: the lower bound of the subview
    /// - Parameter upper: the upper bound of the subview
    /// - Returns: the extents to be used to create a subview
    @inlinable
    func getExtents(from lower: Shape.Array, to upper: Shape.Array)
        -> Shape.Array
    {
        // bounds should be in the correct order by the time they reach here
        assert({
            for (l, u) in zip(lower, upper) { if l > u { return false } }
            return true
        }(), "lower must be less than or equal to upper")

        var extents = upper
        zip(upper, lower).map(into: &extents, -)
        return extents
    }

    //--------------------------------------------------------------------------
    /// getExtents(_:_:_
    /// computes the extents and strides from the specified bounds and steps
    /// - Parameter lower: the lower bound of the subview
    /// - Parameter upper: the upper bound of the subview
    /// - Parameter steps: the step interval along each dimension. This
    ///                    value can be negative to perform reverse traversal
    /// - Returns: the extents and strides to be used to create a subview
    @inlinable
    func getExtents(_ lower: Shape.Array,
                    _ upper: Shape.Array,
                    _ steps: Shape.Array) ->
        (extents: Shape.Array, strides: Shape.Array)
    {
        // if all the steps are 1, then just reuse the parent strides
        if steps.first(where: { $0 != 1 }) == nil {
            return (getExtents(from: lower, to: upper), self.strides)

        } else {
            // if one or more steps are not 1,
            // then recompute the subview extents and strides

            // y must be positive for this to work correctly
            func divceil(_ x: Int, _ y: Int) -> Int { (x - 1 + y) / y }
            
            var subExtents = getExtents(from: lower, to: upper)
            zip(subExtents, steps).map(into: &subExtents) {
                divceil($0, Swift.abs($1))
            }
            
            var subStrides = strides
            zip(strides, steps).map(into: &subStrides, *)
            return (subExtents, subStrides)
        }
    }

    //--------------------------------------------------------------------------
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(lower: Shape.Tuple, upper: Shape.Tuple, steps: Shape.Tuple)
        -> Self {
        get { self[Shape.Array(lower), Shape.Array(upper), Shape.Array(steps)] }
        set {
            self[Shape.Array(lower), Shape.Array(upper),
                 Shape.Array(steps)] = newValue
        }
    }
    
    //--------------------------------------------------------------------------
    // views will have the same shared state as the parent
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(lower: Shape.Array, upper: Shape.Array,
              steps: Shape.Array) -> Self
    {
        get {
            let (extents, strides) = getExtents(lower, upper, steps)
            return view(at: lower, extents: extents, strides: strides)
        }
        set {
            // make sure a possibly repeated view is dense before getting
            // the extents and strides used for writing.
            makeDense(view: self)
            let (extents, strides) = getExtents(lower, upper, steps)
            var view = mutableView(at: lower, extents: extents, strides: strides)
            copy(from: newValue, to: &view)
        }
    }
    
    //--------------------------------------------------------------------------
    /// makeDense(view:
    /// if the view is already dense, then noop. If the view is repeated,
    /// then the view virtual elements are realized and the view is converted
    /// to dense.
    // Note: This is not part of mutableView, because there are cases
    // where we want to interact with a repeated view, such as in reductions
    @inlinable
    mutating func makeDense(view: Self) {
        guard shape.spanCount < shape.count else { return }

        // create storage for all elements
        var dense = createDense()
        
        // report
        diagnostic("\(realizeString) \(name)(\(tensorArray.trackingId)) " +
            "expanding from: \(String(describing: Element.self))" +
            "[\(shape.spanCount)] " +
            "to: \(String(describing: Element.self))[\(dense.count)]",
            categories: [.dataRealize, .dataCopy])
        
        // perform and indexed copy and assign to self
        currentQueue.copy(from: self, to: &dense)
        self = dense
    }
}

//==============================================================================
/// Derivative registration
extension TensorView where Self: DifferentiableTensorView {
    // https://github.com/apple/swift/blob/37b507b31c77ef969151f385cd1902dd44fb3b7f/stdlib/public/core/Array.swift#L2091
    @inlinable
    @derivative(of: subscript)
    func _vjpSubscript(lower: Shape.Array, upper: Shape.Array, steps: Shape.Array)
        -> (value: Self, pullback: (Self) -> Self)
    {
        return (self[lower, upper, steps], { v in
            var result = self.filled(with: Element())
            result[lower, upper, steps] = v
            return result
        })
    }
}

//==============================================================================
public protocol TensorIndexing: Strideable {
    associatedtype Position
    /// sequential logical view element index
    var viewIndex: Int { get }
    /// linear data buffer element index
    var dataIndex: Int { get }

    /// initializer for starting at any position
    init<T>(view: T, at position: Position) where T: TensorView
    /// initializer specifically for the endIndex
    init<T>(endOf view: T) where T: TensorView
    
    /// highest frequency function to move the index
    /// use advanced(by n: for jumps or negative movement
    func increment() -> Self
}

public extension TensorIndexing {
    // Equatable
    @inlinable
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.viewIndex == rhs.viewIndex
    }
    
    // Comparable
    @inlinable
    static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.viewIndex < rhs.viewIndex
    }
    
    @inlinable
    func distance(to other: Self) -> Int { other.viewIndex - viewIndex }
}

//==============================================================================
/// TensorValueCollection
public struct TensorValueCollection<View>: RandomAccessCollection
    where View: TensorView
{
    // properties
    public let view: View
    public let buffer: UnsafeBufferPointer<View.Element>
    public let startIndex: View.Index
    public let endIndex: View.Index
    public let count: Int

    @inlinable
    public init(view: View, buffer: UnsafeBufferPointer<View.Element>) {
        self.view = view
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.count
    }

    @inlinable
    public init(view: View) {
        self.view = view
        self.buffer = UnsafeBufferPointer<View.Element>(start: nil, count: 0)
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.count
    }
    
    //--------------------------------------------------------------------------
    // Collection
    @inlinable
    public func index(before i: View.Index) -> View.Index { i.advanced(by: -1) }
    
    @inlinable
    public func index(after i: View.Index) -> View.Index { i.increment() }
    
    @inlinable
    public subscript(index: View.Index) -> View.Element {
        buffer[index.dataIndex]
    }
}

//==============================================================================
/// TensorMutableValueCollection
public struct TensorMutableValueCollection<View>: RandomAccessCollection,
    MutableCollection where View: TensorView
{
    // properties
    public let buffer: UnsafeMutableBufferPointer<View.Element>
    public let startIndex: View.Index
    public let endIndex: View.Index
    public let count: Int
    
    @inlinable
    public init(view: inout View,
                buffer: UnsafeMutableBufferPointer<View.Element>) {
        self.buffer = buffer
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.count
    }
    
    @inlinable
    public init(view: inout View) {
        self.buffer = UnsafeMutableBufferPointer<View.Element>(start: nil,
                                                               count: 0)
        startIndex = view.startIndex
        endIndex = view.endIndex
        count = view.count
    }
    
    //--------------------------------------------------------------------------
    // Collection
    @inlinable
    public func index(before i: View.Index) -> View.Index { i.advanced(by: -1) }
    
    @inlinable
    public func index(after i: View.Index) -> View.Index { i.increment() }
    
    @inlinable
    public subscript(index: View.Index) -> View.Element {
        get {
            buffer[index.dataIndex]
        }
        set {
            buffer[index.dataIndex] = newValue
        }
    }
}
