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
/// DType
/// the implicit tensor Element type
public typealias DType = Float

//==============================================================================
/// FillTensor
public struct FillTensor<Shape, Element>: Tensor where Shape: Shaped {
    // properties
    @inlinable public static var name: String { "FillTensor\(Shape.rank)" }
    public let elementCount: Int
    public let shape: Shape
    public let storageOrder: StorageOrder
    public let element: Element
    
    @inlinable
    public init(
        _ shape: Shape,
        element: Element,
        order: StorageOrder = .rowMajor
    ) {
        self.elementCount = shape.elementCount()
        self.shape = shape
        self.storageOrder = order
        self.element = element
    }
    
    @inlinable
    public func elements() -> FillTensorIterator<Shape, Element> {
        FillTensorIterator(elementCount, element)
    }
    
    @inlinable
    public subscript(position: Shape, shape: Shape) -> Self {
        FillTensor(shape, element: element, order: storageOrder)
    }
    
    @inlinable
    public subscript(position: Shape, shape: Shape, steps: Shape) -> Self {
        FillTensor(shape, element: element, order: storageOrder)
    }
}

//==============================================================================
/// FillTensorIterator
public struct FillTensorIterator<Shape, Element>: Sequence, IteratorProtocol
    where Shape: Shaped
{
    public var count: Int
    public let element: Element
    
    @inlinable public init(_ count: Int, _ element: Element) {
        self.count = count
        self.element = element
    }

    @inlinable public mutating func next() -> Element? {
        guard count > 0 else { return nil }
        count -= 1
        return element
    }
}

//==============================================================================
/// EyeTensor
public struct EyeTensor<Element>: Tensor where Element: Numeric {
    // properties
    @inlinable public static var name: String { "EyeTensor" }
    public let elementCount: Int
    public let shape: Shape2
    public let storageOrder: StorageOrder
    public let k: Int

    @inlinable
    public init(
        _ N: Int, _ M: Int, _ k: Int,
        _ order: StorageOrder = .rowMajor
    ) {
        self.k = k
        self.shape = Shape2(N, M)
        self.elementCount = N * M
        self.storageOrder = order
    }
    
    @inlinable
    public func elements() -> EyeTensorIterator<Element> {
        EyeTensorIterator(shape, k)
    }

    public subscript(position: Shape2, shape: Shape2) -> Self {
        EyeTensor(shape[0], shape[1], k)
    }
    
    public subscript(position: Shape2, shape: Shape2, steps: Shape2) -> Self {
        EyeTensor(shape[0], shape[1], k)
    }
}

//==============================================================================
/// EyeTensorIterator
public struct EyeTensorIterator<Element>: Sequence, IteratorProtocol
    where Element: Numeric
{
    public let shape: Shape2
    public var position: Shape2
    public let k: Shape2

    @inlinable public init(_ shape: Shape2, _ k: Int) {
        self.shape = shape
        position = Shape2.zero
        self.k = k < 0 ? Shape2(-k, 0) : Shape2(0, k)
    }

    @inlinable public mutating func next() -> Element? {
        // when the row position equals the number of rows, then we're done
        guard position[0] != shape[0] else { return nil }
        defer { position.increment(boundedBy: shape) }
        
        // if the axes indexes are equal then it's on the diagonal
        let pos = position &- k
        return pos[0] == pos[1] ? 1 : 0
    }
}

//==============================================================================
/// DenseTensor
public struct DenseTensor<Shape, Element>:
    MutableTensor, MutableCollection where Shape: Shaped
{
    public typealias Index = SequentialIndex<Shape>

    public let storageBuffer: TensorBuffer<Element>
    /// the dense number of elements in the shape
    public let elementCount: Int
    /// the linear element offset where the view begins
    public let bufferOffset: Int
    /// Specifies whether data is stored in row-major (C-style)
    /// or column-major (Fortran-style) order in memory.
    public let storageOrder: StorageOrder
    /// the dimensions of the element space
    public let shape: Shape
    /// `true` if the view will be shared by by multiple writers
    public let isShared: Bool
    /// The strided number of elements spanned by the shape
    public let spanCount: Int
    /// The distance to the next element along each dimension
    public let strides: Shape

    @inlinable public static var name: String { "DenseTensor\(Shape.rank)" }
    @inlinable public var startIndex: Index { Index(Shape.zero, 0) }
    @inlinable public var endIndex: Index { Index(Shape.zero, elementCount) }

    //--------------------------------------------------------------------------
    /// init(shape:
    @inlinable
    public init(
        _ shape: Shape,
        storageBuffer: TensorBuffer<Element>? = nil,
        strides: Shape? = nil,
        bufferOffset: Int = 0,
        share: Bool = false,
        order: StorageOrder = .rowMajor
    ) {
        let elementCount = shape.elementCount()
        self.storageBuffer = storageBuffer ??
            TensorBuffer(count: elementCount, name: Self.name)
        self.elementCount = elementCount
        self.bufferOffset = bufferOffset
        self.storageOrder = order
        self.shape = shape
        self.isShared = share
        let sequentialStrides = shape.sequentialStrides()

        if let callerStrides = strides {
            self.strides = callerStrides
            self.spanCount =  ((shape &- 1) &* callerStrides).wrappedSum() + 1
        } else {
            self.strides = sequentialStrides
            self.spanCount = elementCount
        }
    }
    
    @inlinable
    public func elements() -> DenseTensorIterator<Shape, Element> {
        DenseTensorIterator(self, startIndex)
    }
    
    @inlinable
    public func mutableElements() -> Self { self }
    
    //--------------------------------------------------------------------------
    //
    @inlinable
    public func shared(at position: Shape, with shape: Shape, by steps: Shape?)
        -> Self
    {
        fatalError()
    }
    
    @inlinable
    public subscript(position: Shape, shape: Shape, steps: Shape) -> Self {
        get {
            fatalError()
        }
        set {
            fatalError()
        }
    }
    
    public subscript(position: Shape, shape: Shape) -> Self {
        get {
            fatalError()
        }
        set {
            fatalError()
        }
    }
    
    //--------------------------------------------------------------------------
    // Collection
    @inlinable
    public func index(after i: Index) -> Index {
        var index = i
        index.position.increment(boundedBy: shape)
        index.sequenceIndex += 1
        return index
    }

    @inlinable
    public subscript(index: Index) -> Element {
        get {
            fatalError()
        }
        set {
            fatalError()
        }
    }
}

//==============================================================================
/// DenseTensorIterator
public struct DenseTensorIterator<Shape, Element>: Sequence, IteratorProtocol
    where Shape: Shaped
{
    public let tensor: DenseTensor<Shape, Element>
    public var index: SequentialIndex<Shape>
    
    @inlinable public init(_ tensor: DenseTensor<Shape, Element>,
                           _ index: SequentialIndex<Shape>) {
        self.tensor = tensor
        self.index = index
    }
    
    @inlinable public mutating func next() -> Element? {
        index.position.increment(boundedBy: tensor.shape)
        index.sequenceIndex += 1
        return tensor[index]
    }
}

//==============================================================================
/// SequentialIndex
public struct SequentialIndex<Shape>: Comparable where Shape: Shaped {
    /// the logical position along each axis
    public var position: Shape
    /// linear sequence position
    public var sequenceIndex: Int

    /// `init(position:sequenceIndex:`
    @inlinable
    public init(_ position: Shape, _ sequenceIndex: Int) {
        self.position = position
        self.sequenceIndex = sequenceIndex
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


////==============================================================================
//// Tensor extensions
//extension Tensor: Equatable where Element: Equatable { }
//extension Tensor: Codable where Element: Codable { }
//
//extension Tensor: Differentiable & DifferentiableTensorView
//    where Element: DifferentiableElement
//{
//    public typealias TangentVector = Tensor
//}
//
//extension Tensor: AdditiveArithmetic where Element: Numeric {
//    @inlinable @_transparent public static var zero: Self { Self(Element.zero) }
//    @inlinable @_transparent public static var one: Self { Self(Element.one) }
//}
//
////==============================================================================
//// Tensor
//public extension Tensor {
//    //--------------------------------------------------------------------------
//    /// reserved space
//    @inlinable
//    init(bounds: Shape, storage order: StorageOrder = .C) {
//        self = Self.create(TensorShape(bounds, storage: order))
//    }
//
//    //--------------------------------------------------------------------------
//    /// repeating element
//    @inlinable
//    init(repeating value: Element, to bounds: Shape.Tuple,
//         storage order: StorageOrder = .C)
//    {
//        let shape = TensorShape(Shape(bounds), strides: Shape.zero, storage: order)
//        self = Self.create(for: value, shape)
//    }
//
//    //--------------------------------------------------------------------------
//    // typed views
//    @inlinable
//    func createBoolTensor(with bounds: Shape) -> Tensor<Shape, Bool> {
//        Tensor<Shape, Bool>(bounds: bounds)
//    }
//
//    @inlinable
//    func createIndexTensor(with bounds: Shape) -> Tensor<Shape, IndexType> {
//        Tensor<Shape, IndexType>(bounds: bounds)
//    }
//}
//
////==============================================================================
//// Tensor1
//public extension Tensor where Shape == Shape1 {
//    // Swift array of elements
//    @inlinable
//    var array: [Element] { [Element](bufferElements()) }
//
//    var description: String { "\(array)" }
//
//    // simplified integer index
//    @inlinable
//    subscript(index: Int) -> Element {
//        get {
//            view(from: makePositive(index: Shape(index)),
//                 to: Shape.one, with: Shape.one).element
//        }
//        set {
//            expandSelfIfRepeated()
//            var view = sharedView(from: makePositive(index: Shape(index)),
//                                  to: Shape.one, with: Shape.one)
//            view.element = newValue
//        }
//    }
//
//    // simplified integer range
//    @inlinable
//    subscript<R>(range: R) -> Self
//        where R: PartialRangeExpression, R.Bound == Int
//        {
//        get {
//            let r = range.relativeTo(0..<bounds[0])
//            return self[Shape(r.start), Shape(r.end), Shape(r.step)]
//        }
//        set {
//            let r = range.relativeTo(0..<bounds[0])
//            self[Shape(r.start), Shape(r.end), Shape(r.step)] = newValue
//        }
//    }
//
//}
//
////==============================================================================
//// Tensor2
//public extension Tensor where Shape == Shape2
//{
//    //--------------------------------------------------------------------------
//    /// Swift array of elements
//    @inlinable
//    var array: [[Element]] {
//        var result = [[Element]]()
//        for row in 0..<bounds[0] {
//            result.append([Element](self[row, ...].bufferElements()))
//        }
//        return result
//    }
//
//    var description: String { "\(array)" }
//
//    //--------------------------------------------------------------------------
//    // subscripting a Matrix view
//    @inlinable
//    subscript<R, C>(rows: R, cols: C) -> Self where
//        R: PartialRangeExpression, R.Bound == Int,
//        C: PartialRangeExpression, C.Bound == Int
//    {
//        get {
//            let r = rows.relativeTo(0..<bounds[0])
//            let c = cols.relativeTo(0..<bounds[1])
//            return self[Shape(r.start, c.start), Shape(r.end, c.end),
//                        Shape(r.step, c.step)]
//        }
//
//        set {
//            let r = rows.relativeTo(0..<bounds[0])
//            let c = cols.relativeTo(0..<bounds[1])
//            self[Shape(r.start, c.start), Shape(r.end, c.end),
//                 Shape(r.step, c.step)] = newValue
//        }
//    }
//
//    @inlinable
//    subscript<R>(rows: R, cols: UnboundedRange) -> Self
//        where R: PartialRangeExpression, R.Bound == Int {
//        get { self[rows, 0...] }
//        set { self[rows, 0...] = newValue }
//    }
//
//    @inlinable
//    subscript<C>(rows: UnboundedRange, cols: C) -> Self
//        where C: PartialRangeExpression, C.Bound == Int {
//        get { self[0..., cols] }
//        set { self[0..., cols] = newValue }
//    }
//
//}
//
