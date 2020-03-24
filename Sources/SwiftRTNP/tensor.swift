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
import SwiftRT

//==============================================================================
/// DType
/// the implicit tensor Element type
public typealias DType = Float

//==============================================================================
///
public typealias Tensor0<Element> = Tensor<Bounds1, Element>
public typealias Tensor1<Element> = Tensor<Bounds1, Element>
public typealias Tensor2<Element> = Tensor<Bounds2, Element>
public typealias Tensor3<Element> = Tensor<Bounds3, Element>
public typealias Tensor4<Element> = Tensor<Bounds4, Element>
public typealias Tensor5<Element> = Tensor<Bounds5, Element>

//==============================================================================
// Tensor
public struct Tensor<Bounds, Element>: TensorView
    where Bounds: ShapeBounds
{
    // properties
    public static var diagnosticName: String { "Tensor\(Bounds.rank)" }
    public let shape: TensorShape<Bounds>
    public var buffer: TensorBuffer<Element>
    public let offset: Int
    public let shared: Bool

    @inlinable
    public init(shape: TensorShape<Bounds>, buffer: TensorBuffer<Element>,
                offset: Int, shared: Bool)
    {
        self.shape = shape
        self.buffer = buffer
        self.offset = offset
        self.shared = shared
    }
}

//==============================================================================
// Tensor extensions
extension Tensor: Equatable where Element: Equatable { }
extension Tensor: Codable where Element: Codable { }

extension Tensor: Differentiable & DifferentiableTensorView
    where Element: DifferentiableElement
{
    public typealias TangentVector = Tensor
}

extension Tensor: AdditiveArithmetic where Element: Numeric {
    @inlinable @_transparent public static var zero: Self { Self(Element.zero) }
    @inlinable @_transparent public static var one: Self { Self(Element.one) }
}

//==============================================================================
// Tensor
public extension Tensor {
    //--------------------------------------------------------------------------
    /// reserved space
    @inlinable
    init(bounds: Bounds, storage order: StorageOrder = .C) {
        self = Self.create(TensorShape(bounds, storage: order))
    }

    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    init(repeating value: Element, to bounds: Bounds.Tuple,
         storage order: StorageOrder = .C)
    {
        let shape = TensorShape(Bounds(bounds), strides: Bounds.zero, storage: order)
        self = Self.create(for: value, shape)
    }

    //--------------------------------------------------------------------------
    // typed views
    @inlinable
    func createBoolTensor(with bounds: Bounds) -> Tensor<Bounds, Bool> {
        Tensor<Bounds, Bool>(bounds: bounds)
    }
    
    @inlinable
    func createIndexTensor(with bounds: Bounds) -> Tensor<Bounds, IndexType> {
        Tensor<Bounds, IndexType>(bounds: bounds)
    }
}

//==============================================================================
// Tensor1
public extension Tensor where Bounds == Bounds1 {
    // Swift array of elements
    @inlinable
    var array: [Element] { [Element](bufferElements()) }

    var description: String { "\(array)" }

    // simplified integer index
    @inlinable
    subscript(index: Int) -> Element {
        get {
            view(from: makePositive(index: Bounds(index)),
                 to: Bounds.one, with: Bounds.one).element
        }
        set {
            expandSelfIfRepeated()
            var view = sharedView(from: makePositive(index: Bounds(index)),
                                  to: Bounds.one, with: Bounds.one)
            view.element = newValue
        }
    }
    
    // simplified integer range
    @inlinable
    subscript<R>(range: R) -> Self
        where R: PartialRangeExpression, R.Bound == Int
        {
        get {
            let r = range.relativeTo(0..<bounds[0])
            return self[Bounds(r.start), Bounds(r.end), Bounds(r.step)]
        }
        set {
            let r = range.relativeTo(0..<bounds[0])
            self[Bounds(r.start), Bounds(r.end), Bounds(r.step)] = newValue
        }
    }

}

//==============================================================================
// Tensor2
public extension Tensor where Bounds == Bounds2
{
    //--------------------------------------------------------------------------
    /// Swift array of elements
    @inlinable
    var array: [[Element]] {
        var result = [[Element]]()
        for row in 0..<bounds[0] {
            result.append([Element](self[row, ...].bufferElements()))
        }
        return result
    }
    
    var description: String { "\(array)" }

    //--------------------------------------------------------------------------
    // subscripting a Matrix view
    @inlinable
    subscript<R, C>(rows: R, cols: C) -> Self where
        R: PartialRangeExpression, R.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int
    {
        get {
            let r = rows.relativeTo(0..<bounds[0])
            let c = cols.relativeTo(0..<bounds[1])
            return self[Bounds(r.start, c.start), Bounds(r.end, c.end),
                        Bounds(r.step, c.step)]
        }
        
        set {
            let r = rows.relativeTo(0..<bounds[0])
            let c = cols.relativeTo(0..<bounds[1])
            self[Bounds(r.start, c.start), Bounds(r.end, c.end),
                 Bounds(r.step, c.step)] = newValue
        }
    }
    
    @inlinable
    subscript<R>(rows: R, cols: UnboundedRange) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get { self[rows, 0...] }
        set { self[rows, 0...] = newValue }
    }
    
    @inlinable
    subscript<C>(rows: UnboundedRange, cols: C) -> Self
        where C: PartialRangeExpression, C.Bound == Int {
        get { self[0..., cols] }
        set { self[0..., cols] = newValue }
    }

}

