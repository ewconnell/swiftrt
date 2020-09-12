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
// String(timeInterval:
extension String {
    @inlinable
    public init(timeInterval: TimeInterval, precision: Int = 2) {
        let remainder = modf(timeInterval).1
        let interval = Int(timeInterval)
        let seconds = interval % 60
        let minutes = (interval / 60) % 60
        let hours = (interval / 3600)
        let remStr = String(format: "%.\(precision)f", remainder)
        self = String(format: "%0.2d:%0.2d:%0.2d.\(remStr)",
                      hours, minutes, seconds)
    }
}

//==============================================================================
/// elapsedTime
/// used to measure and log a set of `body` iterations
@discardableResult
public func elapsedTime(_ logLabel: String? = nil, iterations: Int = 10,
                        warmUps: Int = 1, precision: Int = 6,
                        _ body: () -> Void) -> TimeInterval
{
    // warm ups are to factor out module or data load times
    if let label = logLabel, warmUps > 0 {
        var warmUpTimings = [TimeInterval]()
        for _ in 0..<warmUps {
            let start = Date()
            body()
            let elapsed = Date().timeIntervalSince(start)
            warmUpTimings.append(elapsed)
        }

        let warmUpAverage = warmUpTimings.reduce(0, +) /
            Double(warmUpTimings.count)
        
        logTimings("\(label) average start up", warmUpTimings,
                   warmUpAverage, precision)
    }
    
    // collect the timings and take the average
    var timings = [TimeInterval]()
    for _ in 0..<iterations {
        let start = Date()
        body()
        let elapsed = Date().timeIntervalSince(start)
        timings.append(elapsed)
    }
    let average = timings.reduce(0, +) / Double(timings.count)

    // log results if requested
    if let label = logLabel {
        logTimings("\(label) average iteration", timings, average, precision)
    }
    return average
}

func logTimings(_ label: String, _ timings: [TimeInterval],
                _ average: TimeInterval, _ precision: Int)
{
    let avgStr = String(timeInterval: average, precision: precision)
    Context.log.write(level: .status, message:
        "\(label) time: \(avgStr)")
    for i in 0..<timings.count {
        let timeStr = String(format: "%.\(precision)f", timings[i])
        Context.log.write(level: .status,
                           message: "Run: \(i) time: \(timeStr)")
    }
    Context.log.write(level: .status, message: "")
}

