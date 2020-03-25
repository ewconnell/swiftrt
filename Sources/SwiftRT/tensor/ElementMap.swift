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
///
public protocol ElementMap {
    /// the order in memory to store materialized Elements
    var order: StorageOrder { get }
}

//==============================================================================
///
public struct FillMap {
    public let order: StorageOrder

    @inlinable public init(order: StorageOrder) {
        self.order = order
    }
}

//==============================================================================
///
public struct DiagonalPatternMap {
    public let order: StorageOrder

    @inlinable public init(order: StorageOrder) {
        self.order = order
    }
}

//==============================================================================
///
public struct SparseMap {
    public let order: StorageOrder

    @inlinable public init(order: StorageOrder) {
        self.order = order
    }
}

//==============================================================================
///
public struct DenseRowMap<Shape>: ElementMap {
    public let order: StorageOrder

    @inlinable public init(_ shape: Shape) {
        order = .rowMajor
    }
}

//==============================================================================
///
public struct DenseColMap: ElementMap {
    public let order: StorageOrder

    @inlinable public init() {
        order = .colMajor
    }
}
