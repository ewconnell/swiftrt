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

@inlinable public func pmap<S0,E0,S1,E1>(
    _ t0: inout Tensor<S0,E0>, axis axis0: Int = 0,
    _ t1: inout Tensor<S1,E1>, axis axis1: Int = 0,
    devices: [Int]? = nil,
    numQueues: Int? = nil,
    _ body: (inout Tensor<S0,E0>, inout Tensor<S1,E1>) -> Void
) {
    body(&t0, &t1)
}
