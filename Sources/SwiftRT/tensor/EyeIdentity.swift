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
/// identity
/// Return the identity array.
/// The identity array is a square array with ones on the main diagonal.
///
/// - Parameters:
///  - n: Int or tuple of Int
///    Shape of the empty array, e.g., (2, 3) or 2.
///  - dtype: data-type, optional
///    Desired output data-type for the array, e.g, Int8. Default is DType.
///  - order: { .C, .F }, optional, default .C
///    Whether to store multi-dimensional data in row-major (C-style)
///    or column-major (Fortran-style) order in memory.
/// - Returns: the identity tensor
@inlinable
public func identity(_ n: Int, order: StorageOrder = .C)
    -> IdentityTensor<DType>
{
    IdentityTensor(n, order)
}

@inlinable
public func identity<Element>(
    _ n: Int,
    _ dtype: Element.Type,
    order: StorageOrder = .C
) -> IdentityTensor<Element> where Element: Numeric
{
    IdentityTensor(n, order)
}
