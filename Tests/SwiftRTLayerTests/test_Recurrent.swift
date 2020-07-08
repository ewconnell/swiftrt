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

class test_Recurrent: XCTestCase {
    // support terminal test run
    static var allTests = [
        ("test_LSTMEncoder", test_LSTMEncoder),
    ]

    //--------------------------------------------------------------------------
    func test_Embedding() {
        let vocabSize = 4
        let encoder = Embedding<Float>(
                vocabularySize: vocabSize,
                embeddingSize: 3,
                embeddingsInitializer: {
                    array(0..<($0[0] * $0[1]), ($0[0], $0[1]))
                })
        let sequence = array([1, 3], type: DeviceIndex.self)
        let embedded = encoder(sequence)
        XCTAssert(embedded == [
            [3.0, 4.0, 5.0],
            [9.0, 10.0, 11.0]
        ])
    }
    
    //--------------------------------------------------------------------------
    func test_LSTMEncoder() {
//        var lstm = LSTM<Float>(LSTMCell(inputSize: 4, hiddenSize: 4))
//        lstm.cell.fusedWeight = lstmInitialCellFusedWeights
//        lstm.cell.fusedBias = lstmInitialCellFusedBias
//
//        let initialState = LSTMCell<Float>.State(cell: lstmInitialStateCell, hidden: lstmInitialStateHidden)
//
//        let outputs = lstm(lstmInputs, initialState: initialState)
//        XCTAssertEqual(outputs.count, 4)
//
//        assertEqual(
//            Tensor(concatenating: outputs.map { $0.hidden }),
//            lstmExpectedStates,
//            accuracy: 1e-6)
//        assertEqual(
//            outputs.last!.cell,
//            lstmExpectedOutput,
//            accuracy: 1e-6)
        
//        let (gradLSTM, gradInputs, gradInitialState) =
//            gradient(at: lstm, lstmInputs, initialState) { lstm, lstmInputs, initialState in
//                lstm.lastOutput(from: lstmInputs, initialState: initialState).cell.sum()
//            }
//
//        assertEqual(
//            gradLSTM.cell.fusedWeight,
//            lstmExpectedGradFusedWeights,
//            accuracy: 1e-6)
//
//        assertEqual(
//            gradLSTM.cell.fusedBias,
//            lstmExpectedGradFusedBias,
//            accuracy: 1e-6)
//
//        assertEqual(
//            Tensor(concatenating: gradInputs.map { $0 }),
//            lstmExpectedGradX,
//            accuracy: 1e-6)
//
//        assertEqual(
//            gradInitialState.cell,
//            lstmExpectedGradInitialStateCell,
//            accuracy: 1e-6)
//        assertEqual(
//            gradInitialState.hidden,
//            lstmExpectedGradInitialStateHidden,
//            accuracy: 1e-6)
    }
}

let lstmInputs =
    [
        array([[ -0.6346513, -0.43788078, -0.40050998, -0.35219777]]),
        array([[-0.59232813,   -0.728384, -0.22974151,   0.3288936]]),
        array([[ 0.5766824,   -0.5468713,  0.16438323,   0.6741958]]),
        array([[-0.47952235, -0.24399251,   0.8553043,   0.8588342]])
    ]

let lstmInitialCellFusedWeights = array(
    [
        [    0.4545238,    0.17410219,    0.07151973,  -0.043370485,     0.4534099,    -0.5096329,
             0.2199418,    -0.2149244,   -0.08141422,   -0.23986903,     0.2063713,    0.17585361,
             0.23440659,    0.43826634,   -0.13891399,   -0.17842606],
        [   -0.3797379,   0.079282284,   -0.10015741,    -0.3239718,     0.2132551,    -0.5461597,
            -0.048002183,    0.26265675,   -0.27132228,    0.39218837,   -0.43364745,   -0.34852988,
            -0.18781787,   -0.41067505,  -0.051611483,     0.4885484],
        [  -0.06933203,    0.54473567,   -0.21874839,   -0.49106207,   -0.13344201,   -0.48908734,
           -0.058430463,   -0.15182033,  -0.071650594,   -0.08455315,     0.5346174,   0.057993293,
           0.03391558,   0.009597003,      0.273346,   -0.49635035],
        [  -0.07164055,    0.37893647,  -0.108646095,    0.07596207,   -0.26968825,     0.4479969,
           -0.085390985,     0.2579115,    0.31213146,    -0.3623113,   -0.20532963,     -0.302337,
           0.2743805,   -0.21505526,   -0.31751555,   -0.44200268],
        [  0.001444459,   -0.44615534,    0.17120321,   0.028076064,    0.18919824,    0.21540813,
           0.21875387,   -0.17696984,   -0.36675194,   -0.37769908,  -0.038331404,    -0.3308463,
           0.24726436,    -0.2989273,    0.26229933,   0.045673575],
        [-0.0060651056,  -0.010614913,   -0.41048288,   -0.16720156,   -0.15950868,  0.0032770582,
         0.02271657,    0.26582226,  -0.042991478,  -0.034523666,   -0.22591552,   -0.46458426,
         -0.38004795,    0.21254668,    0.35087258,    0.35231543],
        [   -0.2352461,   -0.12130469,   -0.19648746,    0.29172993,    0.34960067,   -0.24624759,
            0.3270614,     0.4467347,    0.10191456,    -0.2919168,    -0.1870388,     0.2183716,
            0.11563631,     0.2551108,    0.06388308,    -0.2366966],
        [  -0.36797202,    0.21800567,   -0.06400205,    0.43761855,   -0.10523385,    0.34244883,
           0.11385839,   -0.15765572,     0.1086247,     0.3239568,   -0.23723324,   -0.07836083,
           0.42096126,    0.08826415,    0.10015068,    0.28478572]
    ])

let lstmInitialCellFusedBias = array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

let lstmInitialStateCell = array([[  -0.905442, -0.95177335, -0.05345559,  -1.0015926]])

let lstmInitialStateHidden = array([[ 0.9577427, -0.7377565, 0.35953724, -0.5784317]])

let lstmExpectedStates = array(
    [
        [ -0.24107721,  -0.17449945,  0.037506927,  -0.30990827],
        [ -0.26881245,  -0.00634476, -0.019400658,   -0.2071909],
        [  -0.2842662,  0.032890536, -0.015112571,  -0.13682269],
        [ -0.31182146,   0.08775673, -0.072591506,  -0.07358349]
    ])

let lstmExpectedOutput = array([[ -0.7036462,  0.19863679, -0.14904207, -0.25543863]])

let lstmExpectedGradFusedWeights = array(
    [
        [     0.0805003,    -0.08275342,     0.03522955,   -0.030312486,    -0.25325578,
              -0.16903764,     -0.5353516,    -0.34931934,     0.11720964,    0.075403154,
              -0.0023639333,     0.10740623,    0.016285714,    0.008904672,   -0.003577555,
              -0.006157877],
        [    0.08940099,   -0.095668815,    0.032129563,   -0.013392255,    -0.78527546,
             -0.5366851,    -0.89843607,    -0.68411684,     0.17761928,     0.07160111,
             -2.9360875e-05,     0.17159098,     0.05064467,   0.0044404604,   0.0010539317,
             0.022477672],
        [   -0.12452475,    0.014374051,    -0.06531093,     0.04207304,     0.12626466,
            0.5719551,      0.0789956,     0.21590734,   -0.018112205,     0.05484745,
            -0.0053864187,   -0.044751693,    0.010621901,   0.0052103405,  -0.0023645863,
            -0.00021189533],
        [   -0.18124531,      0.0705468,    -0.08742682,    0.046577547,      0.6233163,
            0.8852248,     0.58292645,     0.62257636,   -0.105407864,    0.021182645,
            -0.0023381193,    -0.14174297,   -0.026014743,   0.0054547014,  -0.0054682447,
            -0.018445207],
        [    0.10980479,   -0.006917909,    0.064796194,       -0.02927,     0.01585731,
             -0.27055886,     0.18191287,   -0.036890566,   -0.072820164,    -0.06949664,
             -0.007851642,  -0.0064141974,   0.0008106679,   -0.012041214,    0.008538902,
             0.004745292],
        [  -0.022427289,   -0.037062984,   -0.021642052,    0.009100203,    -0.31279862,
           -0.057589024,     -0.4670751,    -0.25402433,     0.11365489,     0.07015832,
           0.004176872,    0.073882736,    0.019137694,    0.008957855,   -0.004623856,
           0.0050643114],
        [   0.015348905,    0.013859384,    0.012477151,  -0.0050131194,    0.123534694,
            0.012106663,     0.19668634,    0.098930426,    -0.05052386,   -0.031698443,
            -0.0024695878,    -0.02991161,   -0.006882819,  -0.0043967483,   0.0024977797,
            -0.0014598906],
        [   0.024811186,    -0.05493941,   0.0018467717,  -0.0019808542,    -0.46687102,
            -0.2663457,     -0.5953469,    -0.40291974,     0.13208668,    0.062329676,
            0.0034385747,    0.106468864,     0.02738668,    0.006683163,  -0.0024504995,
            0.010666026]
    ])

let lstmExpectedGradFusedBias = array(
    [   -0.2057409,    0.18275243,  -0.076810926,   0.042995572,     1.5545796,     1.3119537,
        1.812458,     1.4128003,    -0.3810982,  -0.123619094,  -0.008955839,   -0.35501346,
        -0.08343395,  -0.010454423, 0.00085022254,  -0.038896702
    ])

let lstmExpectedGradX = array(
    [[  0.21591075,   0.11293747,  -0.13014226, -0.022586256],
     [  0.08480768,   0.13718912,  -0.12702335,  0.077232406],
     [  0.01666388, 0.0068455637,  -0.35854548,    0.1287557],
     [  -0.2761167, -0.088074416,    -0.415294,   0.32159615]
    ])

let lstmExpectedGradInitialStateCell = array(
    [[0.48873883, 0.23135301,  0.6608742, 0.37556332]])

let lstmExpectedGradInitialStateHidden = array(
    [[ 0.25400645,  0.07120966,  0.36865664, -0.05423181]])
