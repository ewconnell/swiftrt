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
import CCuda

//==============================================================================
/// MatmulProperties
public struct MatmulProperties {

}

//==============================================================================
/// queryMatmulProperties
public func queryMatmulProperties<E>(
    _ a: TensorR2<E>, 
    _ transA: Bool,
    _ b: TensorR2<E>,
    _ transB: Bool,
    _ c: inout TensorR2<E>
) -> MatmulProperties {


    return MatmulProperties()
}