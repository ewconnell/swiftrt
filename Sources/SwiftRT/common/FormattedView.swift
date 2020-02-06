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
// TensorView default implementation
public extension TensorView where Element: AnyConvertable & CVarArg {
    //--------------------------------------------------------------------------
    // formatted
    // *** This function is a temporary placeholder and should be replaced with
    // a fully featured formatting function
    func formatted(
        _ scalarFormat: (width: Int, precision: Int)? = nil,
        maxCols: Int = 10,
        maxItems: [Int]? = nil) -> String
    {
        guard !shape.isEmpty else { return "[Empty]\n" }
        var string = ""
        var index = [Int](repeating: 0, count: shape.rank)
        var itemCount = 0
        let indentSize = "  "
        let extents = self.extents.array
        let lastDimension = shape.lastDimension
        let values = self.elements()
        var iterator = values.makeIterator()

        // clamp ranges
        let maxItems = maxItems?.enumerated().map {
            $1 <= extents[$0] ? $1 : extents[$0]
        } ?? extents

        // set header
        string += "\nTensor extents: \(shape.extents.description)\n"

        func appendFormatted(value: Element) {
            let str = String(format: Element.formatString(scalarFormat), value)
            string += "\(str) "
        }

        // recursive rank > 1 formatting
        func format(dim: Int, indent: String) {
            // print the heading unless it's the last two which we print
            // 2d matrix style
            if dim == lastDimension - 1 {
                let header = "at index: \(String(describing: index))"
                string += "\(indent)\(header)\n\(indent)"
                string += String(repeating: "-", count: header.count) + "\n"
                let maxCol = extents[lastDimension] - 1
                let lastCol = maxItems[lastDimension] - 1

                for _ in 0..<maxItems[lastDimension - 1] {
                    string += indent
                    for col in 0...lastCol {
                        if let value = iterator.next() {
                            appendFormatted(value: value)
                            if col == lastCol {
                                string += (col < maxCol) ? " ...\n" : "\n"
                            }
                        }
                    }
                }
                string += "\n\n"

            } else {
                for _ in 0..<maxItems[dim] {
                    // output index header
                    let header = indent +
                    "at index: \(String(describing: index))"
                    string += "\(indent)\(header)\n\(indent)"
                    string += String(repeating: "=", count: header.count) + "\n"

                    // recursively call next contained dimension
                    format(dim: dim + 1, indent: indent + indentSize)
                    index[dim] += 1
                }
            }
        }

        // format based on rank
        switch shape.rank {
        case 0, 1:
            if shape.isScalar {
                let value = iterator.next()!
                appendFormatted(value: value)
                string += "\n"
            } else {
                var col = 0
                while let value = iterator.next(), itemCount < maxItems[0] {
                    appendFormatted(value: value)
                    itemCount += 1
                    col += 1
                    if col == maxCols {
                        string += "\n"
                        col = 0
                    }
                }
            }
            string += "\n"

        default:
            format(dim: 0, indent: "")
            string = String(string.dropLast())
        }

        return string
    }
}
