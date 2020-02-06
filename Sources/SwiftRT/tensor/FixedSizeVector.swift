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
/// conformance indicates that scalar components are of the same type and
/// densely packed. This is necessary for zero copy view type casting of
/// short vector Element types.
/// For example: Matrix<RGBA<Float>> -> NHWCTensor<Float>
///
public protocol FixedSizeVector: Equatable {
    associatedtype Scalar: Equatable & Codable
    static var count: Int { get }
}

public extension FixedSizeVector {
    @inlinable
    static var count: Int {
        MemoryLayout<Self>.size / MemoryLayout<Scalar>.size
    }
}

//==============================================================================
// RGB
public protocol RGBProtocol: FixedSizeVector, AnyElement, Codable {
    var r: Scalar { get set }
    var g: Scalar { get set }
    var b: Scalar { get set }
    init(_ r: Scalar, _ g: Scalar, _ b: Scalar)
}

public extension RGBProtocol {
    // useful discussion on Codable
    // https://www.raywenderlich.com/3418439-encoding-and-decoding-in-swift
    @inlinable
    func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(r)
        try container.encode(g)
        try container.encode(b)
    }

    @inlinable
    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        try self.init(container.decode(Scalar.self),
                      container.decode(Scalar.self),
                      container.decode(Scalar.self))
    }
}

public struct RGB<Scalar>: RGBProtocol
    where Scalar: TensorElementConformance & Numeric
{
    public var r, g, b: Scalar

    @inlinable
    public init() { r = Scalar.zero; g = Scalar.zero; b = Scalar.zero }

    @inlinable
    public init(_ r: Scalar, _ g: Scalar, _ b: Scalar) {
        self.r = r; self.g = g; self.b = b
    }
}

//==============================================================================
// RGBA
public protocol RGBAProtocol: FixedSizeVector, AnyElement, Codable {
    var r: Scalar { get set }
    var g: Scalar { get set }
    var b: Scalar { get set }
    var a: Scalar { get set }
    init(_ r: Scalar, _ g: Scalar, _ b: Scalar, _ a: Scalar)
}

public extension RGBAProtocol {
    @inlinable
    func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(r)
        try container.encode(g)
        try container.encode(b)
        try container.encode(a)
    }
    
    @inlinable
    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        try self.init(container.decode(Scalar.self),
                      container.decode(Scalar.self),
                      container.decode(Scalar.self),
                      container.decode(Scalar.self))
    }
}

public struct RGBA<Scalar> : RGBAProtocol
    where Scalar: Numeric & Codable & Equatable
{
    public var r, g, b, a: Scalar

    @inlinable
    public init() {
        r = Scalar.zero; g = Scalar.zero; b = Scalar.zero; a = Scalar.zero
    }
    
    @inlinable
    public init(_ r: Scalar, _ g: Scalar, _ b: Scalar, _ a: Scalar) {
        self.r = r; self.g = g; self.b = b; self.a = a
    }
}

//==============================================================================
// Stereo
public protocol StereoProtocol: FixedSizeVector, AnyElement, Codable {}

public struct Stereo<Scalar>: StereoProtocol
    where Scalar: Numeric & Codable & Equatable
{
    public var left, right: Scalar

    @inlinable
    public init() { left = Scalar.zero; right = Scalar.zero }

    @inlinable
    public init(_ left: Scalar, _ right: Scalar) {
        self.left = left; self.right = right
    }
}

