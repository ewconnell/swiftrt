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
    @inlinable
    var description: String {
        var string = ""
        let tab = 2
        var rowShape = Shape.one
        rowShape[Shape.rank - 1] = shape[Shape.rank - 1]

        switch Shape.rank {
        case 1:
            let row = [Element](self[Shape.zero, rowShape].elements())
            string += "\(row)"
            
        case 2:
            string += "[\n"
            for _ in 0..<shape[0] {
                let row = [Element](self[Shape.zero, rowShape].elements())
                string.append("\(String(repeating: " ", count: tab))\(row),\n")
            }
            string = String(string.dropLast(2))
            string += "\n]"
            
        default:
            let rowDim = Shape.rank - 2
            var pos = Shape.zero
            
            func addRows(_ dim: Int) {
                let indent = String(repeating: " ", count: dim * tab)
                if dim < rowDim {
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
                    for _ in 0..<shape[dim] {
                        let row = [Element](self[Shape.zero, rowShape].elements())
                        string += "\(indent)\(row),\n"
                    }
                }
                string = String(string.dropLast(2)) + "\n"
            }
            addRows(0)
        }
        return string
    }
}

