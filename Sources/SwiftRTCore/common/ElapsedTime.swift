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
        let remStr = String(format: "%.\(precision)f", remainder).dropFirst()
        self = String(format: "%0.2d:%0.2d:%0.2d\(remStr)",
                      hours, minutes, seconds)
    }
}

//==============================================================================
/// elapsedTime
/// used to measure and log a set of `body` iterations
@discardableResult public func elapsedTime(
    _ logLabel: String = #function,
    runs: Int = 1,
    warmUp: Bool = true,
    precision: Int = 5,
    _ body: () -> Void) -> TimeInterval
{
    // warmup are to factor out module or data load times
    if warmUp {
        let start = Date()
        body()
        let elapsed = Date().timeIntervalSince(start)
        let str = String(timeInterval: elapsed, precision: precision)
        let message = "Elapsed: \(logLabel)  warmup \(str)"
        log.write(level: .status, message: message)
    }
    
    // collect the timings and take the average
    var timings = [TimeInterval]()
    for _ in 0..<runs {
        let start = Date()
        body()
        let elapsed = Date().timeIntervalSince(start)
        timings.append(elapsed)
    }
    let average = timings.reduce(0, +) / Double(timings.count)

    // log results if requested
    logTimings("Elapsed: \(logLabel) average", timings, average, precision)
    return average
}

func logTimings(
    _ label: String,
    _ timings: [TimeInterval],
    _ average: TimeInterval,
    _ precision: Int
) {
    let avgStr = String(timeInterval: average, precision: precision)
    log.write(level: .status, message: "\(label) \(avgStr)")
    
    for i in 0..<timings.count {
        let timeStr = String(timeInterval: timings[i], precision: precision)
        log.write(level: .status,
                          message: "run: \(i) time: \(timeStr)",
                          nestingLevel: 1)
    }
    log.write(level: .status, message: "")
}

