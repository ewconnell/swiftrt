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

//==============================================================================
// map
public extension TensorView {
    //--------------------------------------------------------------------------
    /// map a tensor into a tensor
    @inlinable
    func map<R: TensorView>(
        into result: inout R,
        _ transform: (Element) -> R.MutableValues.Element)
    {
        let values = elements()
        var results = result.mutableElements()
        zip(values.indices, results.indices).forEach {
            results[$1] = transform(values[$0])
        }
    }

    //--------------------------------------------------------------------------
    /// map a tensor to a new tensor
    @inlinable
    func map(_ transform: (Element) -> Element) -> Self {
        var result = createDense()
        let values = elements()
        var results = result.mutableElements()
        zip(values.indices, results.indices).forEach {
            results[$1] = transform(values[$0])
        }
        return result
    }

    //--------------------------------------------------------------------------
    /// reduce to a multi-dimensional tensor
    /// result must have the same extents as `self`, but the actual storage
    /// can be less via strides for reduction dimensions
    @inlinable
    func reduce<T>(
        into result: inout T,
        _ nextPartialResult: (Element, Element) -> Element)
        where
        T: TensorView, Self.Element == T.Element,
        Self.Shape == T.Shape
    {
        assert(extents == result.extents, _messageElementCountMismatch)
        let elts = elements()
        var res = result.mutableElements()
        zip(res.indices, elts.indices).forEach {
            res[$0] = nextPartialResult(res[$0], elts[$1])
        }
    }
    
    //--------------------------------------------------------------------------
    /// reduce to a mutable collection
    @inlinable
    func reduce(
        _ initialResult: Element,
        _ nextPartialResult: (Element, Element) -> Element) -> Self
    {
        var partial = initialResult
        elements().forEach { partial = nextPartialResult(partial, $0) }

        var result = createDense()
        var collection = result.mutableElements()
        collection[collection.startIndex] = partial
        return result
    }

}

//==============================================================================
// map
public extension Sequence {
    /// map a sequence to a tensor
    @inlinable
    func map<R>(into result: inout R,
                _ transform: (Element) -> R.MutableValues.Element) throws where
        R: TensorView
    {
        var iterator = self.makeIterator()
        var results = result.mutableElements()
        
        for i in results.indices {
            if let value = iterator.next() {
                results[i] = transform(value)
            }
        }
    }
    
    /// map to a mutable collection
    @inlinable
    func map<R>(into result: inout R,
                _ transform: (Element) -> R.Element) where
        R: MutableCollection
    {
        
        var iterator = self.makeIterator()
        for i in result.indices {
            if let value = iterator.next() {
                result[i] = transform(value)
            }
        }
    }
}

//==============================================================================
public extension Zip2Sequence {
    typealias Pair = (Sequence1.Element, Sequence2.Element)
    
    /// map tensors
    @inlinable
    func map<T>(into result: inout T,
                _ transform: (Pair) -> T.MutableValues.Element)
        where T: TensorView
    {
        var iterator = self.makeIterator()
        var results = result.mutableElements()
        
        for i in results.indices {
            if let pair = iterator.next() {
                results[i] = transform(pair)
            }
        }
        
    }
    
    /// map to a mutable collection
    @inlinable
    func map<Result>(into result: inout Result,
                     _ transform: (Pair) -> Result.Element)
        where Result: MutableCollection
    {
        var iterator = self.makeIterator()
        for i in result.indices {
            if let pair = iterator.next() {
                result[i] = transform(pair)
            }
        }
    }
}

//==============================================================================
public extension Zip3Sequence {
    typealias Input = (S1.Element, S2.Element, S3.Element)
    
    /// map tensors
    @inlinable
    func map<T>(into result: inout T,
                _ transform: (Input) -> T.MutableValues.Element)
        where T: TensorView
    {
        var iterator = self.makeIterator()
        var results = result.mutableElements()
        
        for i in results.indices {
            if let input = iterator.next() {
                results[i] = transform(input)
            }
        }
        
    }
    
    /// map to a mutable collection
    @inlinable
    func map<Result>(into result: inout Result,
                     _ transform: (Input) -> Result.Element)
        where Result: MutableCollection
    {
        var iterator = self.makeIterator()
        for i in result.indices {
            if let input = iterator.next() {
                result[i] = transform(input)
            }
        }
    }
}

//==============================================================================
// zip
@inlinable
public func zip<T1, T2>(_ t1: T1, _ t2: T2) ->
    Zip2Sequence<TensorValueCollection<T1>, TensorValueCollection<T2>>
    where T1: TensorView, T2: TensorView
{
    zip(t1.elements(), t2.elements())
}

//==============================================================================
// zip
@inlinable
public func zip<T1, T2, T3>(_ t1: T1, _ t2: T2, _ t3: T3) ->
    Zip3Sequence<
    TensorValueCollection<T1>,
    TensorValueCollection<T2>,
    TensorValueCollection<T3>>
    where T1: TensorView, T2: TensorView
{
    zip(t1.elements(), t2.elements(), t3.elements())
}

//==============================================================================
// reduce
public extension Sequence {
    /// reduce to a tensor
    @inlinable
    func reduce<T>(
        into result: inout T,
        _ initialResult: Element,
        _ nextPartialResult: (Element, Element) -> Element)
        where T: TensorView, Element == T.Element
    {
        var results = result.mutableElements()
        var partial = initialResult
        for value in self {
            partial = nextPartialResult(partial, value)
        }
        results[results.startIndex] = partial
    }
    
    //--------------------------------------------------------------------------
    /// reduce to a mutable collection
    @inlinable
    func reduce<T>(
        into result: inout T,
        _ initialResult: Element,
        _ nextPartialResult: (Element, Element) -> Element)
        where T: MutableCollection, Element == T.Element
    {
        var partial = initialResult
        for value in self {
            partial = nextPartialResult(partial, value)
        }
        result[result.startIndex] = partial
    }
}
