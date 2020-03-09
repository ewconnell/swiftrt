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
// VectorView protocol
public protocol VectorView: TensorView where Shape == Shape1 { }

// VectorView initialization extensions
public extension VectorView {
    //--------------------------------------------------------------------------
    /// reserved space
    @inlinable
    init(bounds: Bounds, name: String? = nil) {
        self = Self.create(Shape(bounds: bounds), name)
    }
    
    @inlinable
    init(count: Int, name: String? = nil) {
        self.init(bounds: Bounds(count), name: name)
    }
    
    //--------------------------------------------------------------------------
    /// from single `Element`
    @inlinable
    init(element: Element, name: String? = nil) {
        self = Self.create(for: element, Shape(Bounds(1)), name)
    }
    
    //--------------------------------------------------------------------------
    /// from single `AnyConvertable`
    @inlinable
    init<T>(with element: T, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        self = Self.create(for: Element(any: element), Shape(Bounds(1)), name)
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    init(repeating value: Element, to bounds: Int, name: String? = nil)
    {
        let shape = Shape(Bounds(bounds), strides: Bounds.zero)
        self = Self.create(for: value, shape, name)
    }

    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    @inlinable
    init<C>(elements: C, name: String? = nil) where
        C: Collection, C.Element == Element
    {
        self = Self.create(elements, Shape(Bounds(elements.count)), name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    @inlinable
    init<C>(with elements: C, name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        self = Self.create(elements.lazy.map { Element(any: $0) },
                           Shape(Bounds(elements.count)), name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only bufferRef
    /// useful for memory mapped databases, or hardware device buffers
    @inlinable
    init(referenceTo bufferRef: UnsafeBufferPointer<Element>, name: String? = nil)
    {
        let shape = Shape(Bounds(bufferRef.count))
        self = Self.create(referenceTo: bufferRef, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read write bufferRef
    /// useful for memory mapped databases, or hardware device buffers
    @inlinable
    init(referenceTo bufferRef: UnsafeMutableBufferPointer<Element>,
         name: String? = nil)
    {
        let shape = Shape(Bounds(bufferRef.count))
        self = Self.create(referenceTo: bufferRef, shape, name)
    }

    //--------------------------------------------------------------------------
    /// repeated(bounds:
    @inlinable
    func repeated(to bounds: Int) -> Self {
        Self(shape: shape.repeated(to: Bounds(bounds)),
             buffer: buffer, offset: offset, shared: shared)
    }

    //--------------------------------------------------------------------------
    // typed views
    @inlinable
    func createBoolTensor(with bounds: Bounds) -> VectorType<Bool> {
        VectorType<Bool>(bounds: bounds)
    }
    
    @inlinable
    func createIndexTensor(with bounds: Bounds) -> VectorType<IndexType> {
        VectorType<IndexType>(bounds: bounds)
    }
    
    //--------------------------------------------------------------------------
    // Swift array of elements
    @inlinable
    var array: [Element] { [Element](bufferElements()) }
}

//==============================================================================
// VectorType
public struct VectorType<Element>: VectorView {
    // properties
    public static var diagnosticName: String { "Vector" }
    public let shape: Shape1
    public var buffer: TensorBuffer<Element>
    public let offset: Int
    public let shared: Bool
    
    @inlinable
    public init(shape: Shape1, buffer: TensorBuffer<Element>,
                offset: Int, shared: Bool)
    {
        self.shape = shape
        self.buffer = buffer
        self.offset = offset
        self.shared = shared
    }
}

//==============================================================================
// VectorType extensions
extension VectorType: Equatable where Element: Equatable { }
extension VectorType: Codable where Element: Codable { }

extension VectorType: CustomStringConvertible {
    public var description: String { "\(self.array)" }
}

extension VectorType: Differentiable & DifferentiableTensorView
    where Element: DifferentiableElement
{
    public typealias TangentVector = VectorType
}

extension VectorType: AdditiveArithmetic where Element: Numeric {
    @inlinable
    public static var zero: VectorType<Element> {
        VectorType<Element>(element: Element.zero)
    }
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// MatrixView protocol
public protocol MatrixView: TensorView where Shape == Shape2 { }

public enum MatrixLayout { case rowMajor, columnMajor }

// MatrixView initialization extensions
public extension MatrixView {
    //--------------------------------------------------------------------------
    /// reserved space
    @inlinable
    init(bounds: Bounds, layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self = Self.create(Self.matrixShape(bounds, layout), name)
    }
    
    @inlinable
    init(_ rows: Int, _ cols: Int, layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        self.init(bounds: Bounds(rows, cols), layout: layout, name: name)
    }

    //--------------------------------------------------------------------------
    /// from single `Element`
    @inlinable
    init(element: Element, name: String? = nil) {
        let shape = Shape(bounds: Bounds.one)
        self = Self.create(for: element, shape, name)
    }

    //--------------------------------------------------------------------------
    /// from single `AnyConvertable`
    @inlinable
    init<T>(with element: T, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(bounds: Bounds.one)
        self = Self.create(for: Element(any: element), shape, name)
    }

    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    init(repeating value: Element, to rows: Int, _ cols: Int,
         name: String? = nil)
    {
        let shape = Shape(Bounds(rows, cols), strides: Bounds.zero)
        self = Self.create(for: value, shape, name)
    }

    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    @inlinable
    init<C>(_ rows: Int , _ cols: Int, elements: C,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let shape = Self.matrixShape(Bounds(rows, cols), layout)
        assert(shape.count == elements.count, _messageElementCountMismatch)
        self = Self.create(elements, shape, name)
    }

    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    @inlinable
    init<C>(_ rows: Int, _ cols: Int, with elements: C,
            layout: MatrixLayout = .rowMajor,
            name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let shape = Self.matrixShape(Bounds(rows, cols), layout)
        assert(shape.count == elements.count, _messageElementCountMismatch)
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 2D `Element` collection
    @inlinable
    init<T>(elements: [[T]], name: String? = nil) where T == Element{
        let shape = Shape(Bounds(elements.count, elements.first!.count))
        self = Self.create(elements.joined(), shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 2D `AnyConvertable` collection
    @inlinable
    init<T>(with elements: [[T]], name: String? = nil)
        where T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(Bounds(elements.count, elements.first!.count))
        let flatElements = elements.joined().lazy.map {
            Element(any: $0)
        }
        self = Self.create(flatElements, shape, name)
    }

    //--------------------------------------------------------------------------
    /// with reference to read only bufferRef
    /// useful for memory mapped databases, or hardware device buffers
    @inlinable
    init(_ rows: Int, _ cols: Int,
         referenceTo bufferRef: UnsafeBufferPointer<Element>,
         layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        let shape = Self.matrixShape(Bounds(rows, cols), layout)
        self = Self.create(referenceTo: bufferRef, shape, name)
    }

    //--------------------------------------------------------------------------
    /// with reference to read write bufferRef
    /// useful for memory mapped databases, or hardware device buffers
    @inlinable
    init(_ rows: Int, _ cols: Int,
         referenceTo bufferRef: UnsafeMutableBufferPointer<Element>,
         layout: MatrixLayout = .rowMajor,
         name: String? = nil)
    {
        let shape = Self.matrixShape(Bounds(rows, cols), layout)
        self = Self.create(referenceTo: bufferRef, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// repeated(rows:cols:
    @inlinable
    func repeated(to rows: Int, _ cols: Int) -> Self {
        Self(shape: shape.repeated(to: Bounds(rows, cols)),
             buffer: buffer, offset: offset, shared: shared)
    }

    //--------------------------------------------------------------------------
    // typed views
    @inlinable
    func createBoolTensor(with bounds: Bounds) -> MatrixType<Bool> {
        MatrixType<Bool>(bounds: bounds)
    }
    
    @inlinable
    func createIndexTensor(with bounds: Bounds) -> MatrixType<IndexType> {
        MatrixType<IndexType>(bounds: bounds)
    }

    //--------------------------------------------------------------------------
    // transpose
    @inlinable
    var t: Self {
        Self.init(shape: shape.transposed(),
                  buffer: buffer, offset: offset, shared: shared)
    }
    
    //--------------------------------------------------------------------------
    // utilities
    @inlinable
    static func matrixShape(_ bounds: Bounds, _ layout: MatrixLayout)
        -> Shape
    {
        let shape = Shape(bounds: bounds)
        return layout == .rowMajor ? shape : shape.columnMajor
    }
}

//==============================================================================
// MatrixView collection extensions
public extension MatrixView
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

    //--------------------------------------------------------------------------
    // single element
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(r: Int, c: Int) -> Element {
        get {
            let lower = makePositive(index: Bounds(r, c))
            return view(from: lower, to: lower &+ 1, with: Bounds.one).element
        }
        set {
            expandSelfIfRepeated()
            let lower = makePositive(index: Bounds(r, c))
            var single = sharedView(from: lower, to: lower &+ 1,
                                    with: Bounds.one)
            single.element = newValue
        }
    }

    //--------------------------------------------------------------------------
    // subscripting a Matrix view
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
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
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(rows: R, cols: UnboundedRange) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get { self[rows, 0...] }
        set { self[rows, 0...] = newValue }
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<C>(rows: UnboundedRange, cols: C) -> Self
        where C: PartialRangeExpression, C.Bound == Int {
        get { self[0..., cols] }
        set { self[0..., cols] = newValue }
    }
}

//==============================================================================
// MatrixType
public struct MatrixType<Element>: MatrixView {
    // properties
    public static var diagnosticName: String { "Matrix" }
    public let shape: Shape2
    public var buffer: TensorBuffer<Element>
    public let offset: Int
    public let shared: Bool

    @inlinable
    public init(shape: Shape2, buffer: TensorBuffer<Element>,
                offset: Int, shared: Bool)
    {
        self.shape = shape
        self.buffer = buffer
        self.offset = offset
        self.shared = shared
    }
}

//==============================================================================
// MatrixType extensions
extension MatrixType: Equatable where Element: Equatable { }
extension MatrixType: Codable where Element: Codable { }

extension MatrixType: CustomStringConvertible {
    public var description: String { "\(self.array)" }
}

extension MatrixType: Differentiable & DifferentiableTensorView
    where Element: DifferentiableElement
{
    public typealias TangentVector = MatrixType
}

extension MatrixType: AdditiveArithmetic where Element: Numeric {
    @inlinable
    public static var zero: MatrixType<Element> {
        MatrixType<Element>(element: Element.zero)
    }
}

//******************************************************************************
//******************************************************************************
//******************************************************************************

//==============================================================================
// VolumeView protocol
public protocol VolumeView: TensorView where Shape == Shape3 {}

// VolumeView extensions
public extension VolumeView
{
    //--------------------------------------------------------------------------
    /// reserved space
    @inlinable
    init(bounds: Bounds, name: String? = nil) {
        self = Self.create(Shape(bounds: bounds), name)
    }
    
    @inlinable
    init(_ deps: Int, _ rows: Int, _ cols: Int, name: String? = nil) {
        self.init(bounds: Bounds(deps, rows, cols), name: name)
    }
    
    //--------------------------------------------------------------------------
    /// from single `Element`
    @inlinable
    init(element: Element, name: String? = nil) {
        let shape = Shape(bounds: Bounds.one)
        self = Self.create([element], shape, name)
    }

    //--------------------------------------------------------------------------
    /// from single `AnyConvertable`
    @inlinable
    init<T>(with element: T, name: String? = nil) where
        T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(bounds: Bounds.one)
        self = Self.create([Element(any: element)], shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// repeating element
    @inlinable
    init(repeating value: Element, to deps: Int, _ rows: Int, _ cols: Int,
         name: String? = nil)
    {
        let shape = Shape(Bounds(deps, rows, cols), strides: Bounds.zero)
        self = Self.create(for: value, shape, name)
    }

    //--------------------------------------------------------------------------
    /// from flat `Element` collection
    @inlinable
    init<C>(_ deps: Int, _ rows: Int, _ cols: Int,
            elements: C, name: String? = nil) where
        C: Collection, C.Element == Element
    {
        let shape = Shape(Bounds(deps, rows, cols))
        assert(shape.count == elements.count, _messageElementCountMismatch)
        self = Self.create(elements, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from flat `AnyConvertable` collection
    @inlinable
    init<C>(_ deps: Int, _ rows: Int, _ cols: Int,
            with elements: C, name: String? = nil) where
        C: Collection, C.Element: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(Bounds(deps, rows, cols))
        assert(shape.count == elements.count, _messageElementCountMismatch)
        self = Self.create(elements.lazy.map { Element(any: $0) }, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 3D `Element` collection
    @inlinable
    init<T>(elements: [[[T]]], name: String? = nil) where T == Element{
        let shape = Shape(Bounds(elements.count,
                                 elements.first!.count,
                                 elements.first!.first!.count))
        let flatElements = elements.joined().joined()
        self = Self.create(flatElements, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// from structred 3D `AnyConvertable` collection
    @inlinable
    init<T>(with elements: [[[T]]], name: String? = nil)
        where T: AnyConvertable, Element: AnyConvertable
    {
        let shape = Shape(Bounds(elements.count,
                                 elements.first!.count,
                                 elements.first!.first!.count))
        let flatElements = elements.joined().joined().lazy.map {
            Element(any: $0)
        }
        self = Self.create(flatElements, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read only bufferRef
    /// useful for memory mapped databases, or hardware device buffers
    @inlinable
    init(_ deps: Int, _ rows: Int, _ cols: Int,
         referenceTo bufferRef: UnsafeBufferPointer<Element>,
         name: String? = nil)
    {
        let shape = Shape(Bounds(deps, rows, cols))
        self = Self.create(referenceTo: bufferRef, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// with reference to read write bufferRef
    /// useful for memory mapped databases, or hardware device buffers
    @inlinable
    init(_ deps: Int, _ rows: Int, _ cols: Int,
         referenceTo bufferRef: UnsafeMutableBufferPointer<Element>,
         name: String? = nil)
    {
        let shape = Shape(Bounds(deps, rows, cols))
        self = Self.create(referenceTo: bufferRef, shape, name)
    }
    
    //--------------------------------------------------------------------------
    /// repeated(rows:cols:
    @inlinable
    func repeated(to deps: Int, _ rows: Int, _ cols: Int) -> Self {
        Self(shape: shape.repeated(to: Bounds(deps, rows, cols)),
             buffer: buffer, offset: offset, shared: shared)
    }

    //--------------------------------------------------------------------------
    // typed views
    @inlinable
    func createBoolTensor(with bounds: Bounds) -> VolumeType<Bool> {
        VolumeType<Bool>(bounds: bounds)
    }
    
    @inlinable
    func createIndexTensor(with bounds: Bounds) -> VolumeType<IndexType> {
        VolumeType<IndexType>(bounds: bounds)
    }
}

//==============================================================================
// MatrixView extensions
public extension VolumeView {
    /// Swift array of elements
    @inlinable
    var array: [[[Element]]] {
        var result = [[[Element]]]()
        for di in 0..<bounds[0] {
            var depth = [[Element]]()
            for ri in 0..<bounds[1] {
                depth.append([Element](self[di, ri, ...].bufferElements()))
            }
            result.append(depth)
        }
        return result
    }

    //--------------------------------------------------------------------------
    // single element
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript(d: Int, r: Int, c: Int) -> Element {
        get {
            self[Bounds(d, r, c),
                 Bounds(d + 1, r + 1, c + 1),
                 Bounds(1, 1, 1)].element
        }
        
        set {
            var single = self[Bounds(d, r, c),
                              Bounds(d + 1, r + 1, c + 1),
                              Bounds(1, 1, 1)]
            single.element = newValue
        }
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, R, C>(deps: D, rows: R, cols: C) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        R: PartialRangeExpression, R.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int
        {
        get {
            let d = deps.relativeTo(0..<bounds[0])
            let r = rows.relativeTo(0..<bounds[1])
            let c = cols.relativeTo(0..<bounds[2])
            return self[Bounds(d.start, r.start, c.start),
                        Bounds(d.end, r.end, c.end),
                        Bounds(d.step, r.step, c.step)]
        }
        
        set {
            let d = deps.relativeTo(0..<bounds[0])
            let r = rows.relativeTo(0..<bounds[1])
            let c = cols.relativeTo(0..<bounds[2])
            self[Bounds(d.start, r.start, c.start),
                 Bounds(d.end, r.end, c.end),
                 Bounds(d.step, r.step, c.step)] = newValue
        }
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<D>(deps: D, rows: UnboundedRange, cols: UnboundedRange) -> Self
        where D: PartialRangeExpression, D.Bound == Int {
        get { self[deps, 0..., 0...] }
        set { self[deps, 0..., 0...] = newValue }
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, R>(deps: D, rows: R, cols: UnboundedRange) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        R: PartialRangeExpression, R.Bound == Int {
        get { self[deps, rows, 0...] }
        set { self[deps, rows, 0...] = newValue }
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<D, C>(deps: D, rows: UnboundedRange, cols: C) -> Self where
        D: PartialRangeExpression, D.Bound == Int,
        C: PartialRangeExpression, C.Bound == Int {
        get { self[deps, 0..., cols] }
        set { self[deps, 0..., cols] = newValue }
    }

    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<R>(deps: UnboundedRange, rows: R, cols: UnboundedRange) -> Self
        where R: PartialRangeExpression, R.Bound == Int {
        get { self[0..., rows, 0...] }
        set { self[0..., rows, 0...] = newValue }
    }
    
    @inlinable
    @differentiable(where Self: DifferentiableTensorView)
    subscript<C>(deps: UnboundedRange, rows: UnboundedRange, cols: C) -> Self
        where C: PartialRangeExpression, C.Bound == Int {
        get { self[0..., 0..., cols] }
        set { self[0..., 0..., cols] = newValue }
    }
}

//==============================================================================
// VolumeType
public struct VolumeType<Element>: VolumeView {
    // properties
    public static var diagnosticName: String { "Volume" }
    public let shape: Shape3
    public var buffer: TensorBuffer<Element>
    public let offset: Int
    public let shared: Bool

    @inlinable
    public init(shape: Shape3, buffer: TensorBuffer<Element>,
                offset: Int, shared: Bool)
    {
        self.shape = shape
        self.buffer = buffer
        self.offset = offset
        self.shared = shared
    }
}

//==============================================================================
// VolumeType extensions
extension VolumeType: Equatable where Element: Equatable { }
extension VolumeType: Codable where Element: Codable { }

extension VolumeType: CustomStringConvertible {
    public var description: String { "\(self.array)" }
}

extension VolumeType: Differentiable & DifferentiableTensorView
    where Element: DifferentiableElement
{
    public typealias TangentVector = VolumeType
}

extension VolumeType: AdditiveArithmetic where Element: Numeric {
    @inlinable
    public static var zero: VolumeType<Element> {
        VolumeType<Element>(element: Element.zero)
    }
}

