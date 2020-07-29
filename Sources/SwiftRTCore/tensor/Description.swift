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

public extension Tensor {
    /// - Returns: a formatted String description of the tensor elements
    @inlinable var description: String {
        let tab = 2
        var string = ""

        usingAppThreadQueue {
            switch Shape.rank {
            // as a vector
            case 1:
                string += "\([Element](self))"
                
            // as a matrix
            case 2:
                // set row range
                var lower = Shape.zero
                var upper = Shape.one
                upper[1] = shape[1]
                string += "[\n"

                for _ in 0..<shape[0] {
                    let row = [Element](self[lower, upper])
                    string.append("\(String(repeating: " ", count: tab))\(row),\n")
                    lower[0] += 1
                    upper[0] += 1
                }
                string = String(string.dropLast(2))
                string += "\n]"
            
            // as all higher ranked tensors
            default:
                var pos = Shape.zero

                func addRows(_ dim: Int) {
                    let indent = String(repeating: " ", count: dim * tab)
                    if dim < Shape.rank-2 {
                        while true {
                            string += "\(indent)["
                            if shape[dim] > 1 { string += "\(pos[dim])" }
                            string += "\n"
                            addRows(dim + 1)
                            string += "\(indent)],\n"

                            // increment position
                            pos[dim] += 1
                            if pos[dim] == shape[dim] {
                                pos[dim] = 0
                                break
                            }
                        }
                    } else {
                        // set row range
                        var lower = pos
                        var upper = pos &+ 1
                        upper[Shape.rank-1] = shape[Shape.rank-1]
                        
                        for _ in 0..<shape[dim] {
                            let row = [Element](self[lower, upper])
                            string += "\(indent)\(row),\n"
                            lower[dim] += 1
                            upper[dim] += 1
                        }
                    }
                    string = String(string.dropLast(2)) + "\n"
                }
                addRows(0)
            }
        }
        return string
    }
}

