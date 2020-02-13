////******************************************************************************
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
import Foundation

//==============================================================================
/// QueueEvent
/// A queue event is a barrier synchronization object that is
/// - created by a `DeviceQueue`
/// - recorded on a queue to create a barrier
/// - waited on by one or more threads for group synchronization
public protocol QueueEvent {
    /// the id of the event for diagnostics
    var id: Int { get }
    /// is `true` if the even has occurred, used for polling
    var occurred: Bool { get }
    /// the last time the event was recorded
    var recordedTime: Date? { get set }

    /// measure elapsed time since another event
    func elapsedTime(since other: QueueEvent) -> TimeInterval?
    /// will block the caller until the timeout has elapsed if one
    /// was specified during init, otherwise it will block forever
    func wait() throws
}

//------------------------------------------------------------------------------
public extension QueueEvent {
    /// elapsedTime
    /// computes the timeinterval between two queue event recorded times
    /// - Parameter other: the other event used to compute the interval
    /// - Returns: the elapsed interval. Will return `nil` if this event or
    ///   the other have not been recorded.
    @inlinable
    func elapsedTime(since other: QueueEvent) -> TimeInterval? {
        guard let time = recordedTime,
            let other = other.recordedTime else { return nil }
        return time.timeIntervalSince(other)
    }
}

//------------------------------------------------------------------------------
/// QueueEventOptions
public struct QueueEventOptions: OptionSet {
    public let rawValue: Int
    public static let timing       = QueueEventOptions(rawValue: 1 << 0)
    public static let interprocess = QueueEventOptions(rawValue: 1 << 1)
    
    @inlinable
    public init() { self.rawValue = 0 }
    
    @inlinable
    public init(rawValue: Int) { self.rawValue = rawValue }
}

public enum QueueEventError: Error {
    case timedOut
}
