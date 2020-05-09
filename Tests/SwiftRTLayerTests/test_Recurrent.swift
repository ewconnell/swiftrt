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
import XCTest
import Foundation
import SwiftRT

let ndim = 3
let sentence = "look there's a boy with his hat"

let initialLSTMCellFusedWeights = array(
        [[ -0.10264638,    0.5724375,   0.27309537,   0.17255026,  -0.47668853,    0.4770379,
           -0.33398324,    0.5318817,   0.14245105,   0.29648793,   0.51910985,   -0.3606615],
         [ -0.12656957,  -0.50218564,   0.23851442,   0.56842005, -0.032798648,   -0.4720018,
           -0.06218469,  -0.48546985,  -0.41528985,   0.19846004,   0.39093566,   -0.3098252],
         [  0.45832455,   -0.3720271,  -0.10000759,  -0.27497253,   -0.5491783,   -0.3697677,
            0.033266246,  -0.46082756,   0.31556267,   0.41334385,  -0.08878037,  -0.38898242],
         [  0.05400765,  -0.41349488,  -0.10776892,   -0.2313253,  0.114721656,   0.40440345,
            0.5117364,   -0.5681195,   -0.5347471,  -0.25832775,   0.43469727, -0.093170345],
         [ -0.24448833,   0.41439235,  -0.40069306,    0.2501257,   0.20772547,  -0.02735293,
           -0.011522472,  -0.47641352,  -0.57396966,    0.5580311,   0.13775468,   -0.4013645],
         [  0.33192658,   0.21302766,  -0.49628606,  0.113090396,   0.27634227,   0.49463975,
            0.103300214,    0.2272774,  -0.07694018,    -0.463426,   0.16690546,    0.2600751]])

class test_Recurrent: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_LSTMEncoder", test_LSTMEncoder),
    ]

    //==========================================================================
    // test_LSTMEncoder
    func test_Embedding() {
//        let alphabet = Alphabet([sentence], eos: "</s>", eow: "</w>", pad: "</pad>")
//        let characterSequence =
//            try! CharacterSequence(alphabet: alphabet, appendingEoSTo: sentence)
//        let encoder = Embedding<Float>(vocabularySize: alphabet.count, embeddingSize: ndim)
//        let embedded = encoder(characterSequence)
    }
    
    //==========================================================================
    // test_LSTMEncoder
    func test_LSTMEncoder() {
        var lstmEncoder = LSTM<Float>(LSTMCell(inputSize: ndim, hiddenSize: ndim))
        XCTAssert(lstmEncoder.cell.fusedWeight.shape == [6, 12])
        print(lstmEncoder.cell.fusedWeight)
        
        // set known weights for reproducability
        lstmEncoder.cell.fusedWeight = initialLSTMCellFusedWeights
        
        print(lstmEncoder.cell.fusedWeight.shape)
        print(lstmEncoder.cell.fusedBias)

    }
}
