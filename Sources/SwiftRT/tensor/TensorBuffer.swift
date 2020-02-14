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
import Foundation

//==============================================================================
/// TensorBufferProtocol
public protocol TensorBufferProtocol {
    associatedtype Element
    /// a handle returned by the platform to an assoicated memory buffer
    var handle: Int { get }
    /// returns a read only `Element` buffer pointer
    var buffer: UnsafeBufferPointer<Element> { get }
}

extension TensorBufferProtocol {
    public var buffer: UnsafeBufferPointer<Element> {
        fatalError()
    }
}

//==============================================================================
/// TensorBuffer
/// reference counted access to a host memory buffer managed by the platform
public class TensorBuffer<Element>: TensorBufferProtocol {
    public let handle: Int

    public init() {
        handle = 0
    }
}

//==============================================================================
/// MutableTensorBufferProtocol
public protocol MutableTensorBufferProtocol: TensorBufferProtocol {
    /// returns a mutable `Element` buffer pointer that will be
    /// fully overritten, so it will not be initialized or synchronized. It
    /// becomes the new master version.
    var mutableBuffer: UnsafeMutableBufferPointer<Element> { get }
    /// returns a mutable `Element` buffer pointer. The associated host buffer
    /// will be synchronized with the latest remote version of the buffer data
    var synchronizedMutableBuffer: UnsafeMutableBufferPointer<Element> { get }
}

extension MutableTensorBufferProtocol {
    public var mutableBuffer: UnsafeMutableBufferPointer<Element> {
        fatalError()
    }
    
    public var synchronizedMutableBuffer: UnsafeMutableBufferPointer<Element> {
        fatalError()
    }
}

//==============================================================================
/// MutableTensorBuffer
/// reference counted access to a host memory buffer managed by the platform
public class MutableTensorBuffer<Element>: MutableTensorBufferProtocol {
    public let handle: Int

    public init() {
        handle = 0
    }
}

