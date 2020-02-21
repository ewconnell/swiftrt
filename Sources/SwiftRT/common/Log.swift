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
import Dispatch

//==============================================================================
// Logging
public protocol _Logging {
    /// the logWriter to write to
    var logWriter: Log { get }
    /// the level of reporting for this node
    var logLevel: LogLevel { get }
    /// the name path of this node for hierarchical structures
    var logNamePath: String { get }
    /// the nesting level within a hierarchical model to aid in
    /// message formatting.
    var logNestingLevel: Int { get }
    
    /// tests if a message will be written to the logWriter
    /// - Parameter level: the level of the message (error, warning, ...)
    func willLog(level: LogLevel) -> Bool
    
    /// writes a message to the logWriter
    /// - Parameter message: the message to write
    /// - Parameter level: the level of the message (error, warning, ...)
    /// - Parameter indent: optional indent level for formatting
    /// - Parameter trailing: a trailing fill character to add to the message
    /// - Parameter minCount: the minimum length of the message. If it exceeds
    ///   the actual message length, then trailing fill is used. This is used
    ///   mainly for creating message partitions i.e. "---------"
    func writeLog(_ message: @autoclosure () -> String,
                  level: LogLevel,
                  indent: Int,
                  trailing: String,
                  minCount: Int)
    
    /// writes a diagnostic message to the logWriter
    /// - Parameter message: the message to write
    /// - Parameter categories: the categories this message applies to
    /// - Parameter indent: optional indent level for formatting
    /// - Parameter trailing: a trailing fill character to add to the message
    /// - Parameter minCount: the minimum length of the message. If it exceeds
    ///   the actual message length, then trailing fill is used. This is used
    ///   mainly for creating message partitions i.e. "---------"
    func diagnostic(_ message: @autoclosure () -> String,
                    categories: LogCategories,
                    indent: Int,
                    trailing: String,
                    minCount: Int)
}

//==============================================================================
// Logging
public extension _Logging {
    //--------------------------------------------------------------------------
    /// writeLog
    @inlinable
    func willLog(level: LogLevel) -> Bool {
        level <= logWriter.level || level <= logLevel
    }
    
    //--------------------------------------------------------------------------
    /// writeLog
    @inlinable
    func writeLog(_ message: @autoclosure () -> String,
                  level: LogLevel = .error,
                  indent: Int = 0,
                  trailing: String = "",
                  minCount: Int = 80)
    {
        guard willLog(level: level) else { return }
        logWriter.write(level: level,
                        message: message(),
                        nestingLevel: indent + logNestingLevel,
                        trailing: trailing, minCount: minCount)
    }
    
    //--------------------------------------------------------------------------
    // diagnostic
    #if DEBUG
    @inlinable
    func diagnostic(_ message: @autoclosure () -> String,
                    categories: LogCategories,
                    indent: Int = 0,
                    trailing: String = "",
                    minCount: Int = 80)
    {
        guard willLog(level: .diagnostic) else { return}
        // if subcategories have been selected on the logWriter object
        // then make sure the caller's category is desired
        if let mask = logWriter.categories?.rawValue,
            categories.rawValue & mask == 0 { return }
        
        logWriter.write(level: .diagnostic,
                        message: message(),
                        nestingLevel: indent + logNestingLevel,
                        trailing: trailing, minCount: minCount)
    }
    #else
    @inlinable
    func diagnostic(_ message: @autoclosure () -> String,
                    categories: LogCategories,
                    indent: Int = 0,
                    trailing: String = "",
                    minCount: Int = 80) { }
    #endif
}


//==============================================================================
/// LogInfo
/// this is used to manage which logWriter to use and message parameters
public struct LogInfo {
    /// the log writing object to use
    public var logWriter: Log
    /// the reporting level of the object, which allows different objects
    /// to have different reporting levels to fine tune output
    public var logLevel: LogLevel
    /// `namePath` is used when reporting from hierarchical structures
    /// such as a model, so that duplicate names such as `weights` are
    /// put into context
    public var namePath: String
    /// the nesting level within a hierarchical model to aid in
    /// message formatting.
    public var nestingLevel: Int
    
    //--------------------------------------------------------------------------
    @inlinable
    public init(logWriter: Log, logLevel: LogLevel,
                namePath: String, nestingLevel: Int) {
        self.logWriter = logWriter
        self.logLevel = logLevel
        self.namePath = namePath
        self.nestingLevel = nestingLevel
    }
    
    //--------------------------------------------------------------------------
    /// a helper to create logging info for a child object in a hierarchy
    @inlinable
    public func child(_ name: String) -> LogInfo {
        LogInfo(logWriter: logWriter, logLevel: .error,
                namePath: "\(namePath)/\(name)", nestingLevel: nestingLevel + 1)
    }

    //--------------------------------------------------------------------------
    /// a helper to create logging info for an object in a flat
    /// reporting structure
    @inlinable
    public func flat(_ name: String) -> LogInfo {
        LogInfo(logWriter: logWriter, logLevel: .error,
                namePath: "\(namePath)/\(name)", nestingLevel: nestingLevel)
    }
}

//==============================================================================
/// Logging
/// this is conformed to by lightweight objects such as tensors that want
/// to make log entries without carrying any logging state information
public protocol Logging : _Logging {}

public extension Logging {
    @inlinable var logWriter: Log { Platform.log }
    @inlinable var logLevel: LogLevel { Platform.log.level }
    @inlinable var logNamePath: String { "" }
    @inlinable var logNestingLevel: Int { 0 }
}

//==============================================================================
/// Logger
/// this is conformed to by objects that have structured state such as an
/// operator graph or device hierarchy
public protocol Logger : Logging {
    var logInfo: LogInfo { get }
}

extension Logger {
    @inlinable public var logWriter: Log { logInfo.logWriter }
    @inlinable public var logLevel: LogLevel { logInfo.logLevel }
    @inlinable public var logNamePath: String { logInfo.namePath }
    @inlinable public var logNestingLevel: Int { logInfo.nestingLevel }
}

//==============================================================================
/// LogWriter
/// implemented by objects that write to a logWriter.
public protocol LogWriter: class {
    /// the diagnostic categories that will be logged. If `nil`,
    /// all diagnostic categories will be logged
    var categories: LogCategories? { get set }
    /// message levels greater than or equal to this will be logged
    var level: LogLevel { get set }
    /// if `true`, messages are silently discarded
    var _silent: Bool { get set }
    /// the tabsize to use for message formatting
    var _tabSize: Int { get set }
    /// A logWriter can be written to freely by any thread, so create write queue
    var queue: DispatchQueue { get }
    
    //--------------------------------------------------------------------------
    /// write
    /// writes an entry into the logWriter
    /// - Parameter level: the level of the message
    /// - Parameter message: the message string to write
    /// - Parameter nestingLevel: formatting nesting level
    /// - Parameter trailing: a trailing fill character to add to the message
    /// - Parameter minCount: the minimum length of the message. If it exceeds
    ///   the actual message length, then trailing fill is used. This is used
    ///   mainly for creating message partitions i.e. "---------"
    func write(level: LogLevel,
               message: @autoclosure () -> String,
               nestingLevel: Int,
               trailing: String,
               minCount: Int)

    //--------------------------------------------------------------------------
    /// output(message:
    /// writes the formatted message to the logWriter
    func output(message: String)
}

//==============================================================================
// LogWriter
public extension LogWriter {
    @inlinable
    var silent: Bool {
        get { return queue.sync { return _silent } }
        set { queue.sync { _silent = newValue } }
    }

    @inlinable
    var tabSize: Int {
        get { return queue.sync { return _tabSize } }
        set { queue.sync { _tabSize = newValue } }
    }
    
    //--------------------------------------------------------------------------
    /// write
    @inlinable
    func write(level: LogLevel,
               message: @autoclosure () -> String,
               nestingLevel: Int = 0,
               trailing: String = "",
               minCount: Int = 0) {
        // protect against mt writes
        queue.sync { [unowned self] in
            guard !self._silent else { return }
            
            // create fixed width string for level column
            let messageStr = message()
            let levelStr = String(describing: level).padding(
                toLength: LogLevel.maxStringWidth, withPad: " ", startingAt: 0)
            
            let indent = String(repeating: " ",
                                count: nestingLevel * self._tabSize)
            var outputStr = levelStr + ": " + indent + messageStr
            
            // add trailing fill if desired
            if !trailing.isEmpty {
                let fillCount = minCount - outputStr.count
                if messageStr.isEmpty {
                    outputStr += String(repeating: trailing, count: fillCount)
                } else {
                    if fillCount > 1 {
                        outputStr += " " + String(repeating: trailing,
                                                  count: fillCount - 1)
                    }
                }
            }
            output(message: outputStr)
        }
    }
}

//==============================================================================
// Log
public final class Log: LogWriter, ObjectTracking {
    // properties
    public var categories: LogCategories?
    public var level: LogLevel
    public var _silent: Bool
    public var _tabSize: Int
    public var trackingId: Int
    public let queue = DispatchQueue(label: "Log.queue")
    public let logFile: FileHandle

    //--------------------------------------------------------------------------
    /// init(url:isStatic:
    /// - Parameter url: the file to write to. If `nil`,
    ///   output will be written to stdout
    /// - Parameter isStatic: if `true`, indicates that the object
    /// will be held statically so it won't be reported as a memory leak
    @inlinable
    public init(url: URL? = nil, isStatic: Bool = true) {
        assert(url == nil || url!.isFileURL, "Log url must be a file URL")
        level = .error
        trackingId = 0
        _silent = false
        _tabSize = 2
        var file: FileHandle?
        if let fileURL = url?.standardizedFileURL {
            let mgr = FileManager()
            if !mgr.fileExists(atPath: fileURL.path) {
                if !mgr.createFile(atPath: fileURL.path, contents: nil) {
                    print("failed to create logWriter file at: \(fileURL.path)")
                }
            }

            do {
                file = try FileHandle(forWritingTo: fileURL)
                file!.truncateFile(atOffset: 0)
            } catch {
                print(String(describing: error))
            }
        }
        logFile = file ?? FileHandle.standardOutput
        trackingId = ObjectTracker.global.nextId
        ObjectTracker.global.register(self, isStatic: isStatic)
    }
    
    @inlinable
    deinit {
        logFile.closeFile()
        ObjectTracker.global.remove(trackingId: trackingId)
    }
    
    @inlinable
    public func output(message: String) {
        let message = message + "\n"
        logFile.write(message.data(using: .utf8)!)
    }
}

//==============================================================================
// LogEvent
public struct LogEvent {
    var level: LogLevel
    var nestingLevel: Int
    var message: String
}

//------------------------------------------------------------------------------
// LogColors
//  http://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
public enum LogColor: String {
    case reset       = "\u{1b}[m"
    case red         = "\u{1b}[31m"
    case green       = "\u{1b}[32m"
    case yellow      = "\u{1b}[33m"
    case blue        = "\u{1b}[34m"
    case magenta     = "\u{1b}[35m"
    case cyan        = "\u{1b}[36m"
    case white       = "\u{1b}[37m"
    case bold        = "\u{1b}[1m"
    case boldRed     = "\u{1b}[1;31m"
    case boldGreen   = "\u{1b}[1;32m"
    case boldYellow  = "\u{1b}[1;33m"
    case boldBlue    = "\u{1b}[1;34m"
    case boldMagenta = "\u{1b}[1;35m"
    case boldCyan    = "\u{1b}[1;36m"
    case boldWhite   = "\u{1b}[1;37m"
}

public func setText(_ text: String, color: LogColor) -> String {
    #if os(Linux)
    return color.rawValue + text + LogColor.reset.rawValue
    #else
    return text
    #endif
}

//------------------------------------------------------------------------------
// LogCategories
public struct LogCategories: OptionSet {
    public init(rawValue: Int) { self.rawValue = rawValue }
    public let rawValue: Int
    public static let dataAlloc    = LogCategories(rawValue: 1 << 0)
    public static let dataCopy     = LogCategories(rawValue: 1 << 1)
    public static let dataMutation = LogCategories(rawValue: 1 << 2)
    public static let dataRealize  = LogCategories(rawValue: 1 << 3)
    public static let initialize   = LogCategories(rawValue: 1 << 4)
    public static let properties   = LogCategories(rawValue: 1 << 5)
    public static let queueAlloc   = LogCategories(rawValue: 1 << 6)
    public static let queueSync    = LogCategories(rawValue: 1 << 7)
    public static let scheduling   = LogCategories(rawValue: 1 << 8)
}

// strings
public let allocString      = "[\(setText("ALLOCATE ", color: .cyan))]"
public let blockString      = "[\(setText("BLOCK    ", color: .red))]"
public let copyString       = "[\(setText("COPY     ", color: .blue))]"
public let createString     = "[\(setText("CREATE   ", color: .cyan))]"
public let mutationString   = "[\(setText("MUTATE   ", color: .blue))]"
public let realizeString    = "[\(setText("REALIZE  ", color: .cyan))]"
public let recordString     = "[\(setText("RECORD   ", color: .cyan))]"
public let referenceString  = "[\(setText("REFERENCE", color: .cyan))]"
public let releaseString    = "[\(setText("RELEASE  ", color: .cyan))]"
public let schedulingString = "\(setText("~~scheduling", color: .yellow))"
public let signaledString   = "[\(setText("SIGNALED ", color: .green))]"
public let syncString       = "[\(setText("SYNC     ", color: .yellow))]"
public let timeoutString    = "[\(setText("TIMEOUT  ", color: .red))]"
public let waitString       = "[\(setText("WAIT     ", color: .red))]"

//------------------------------------------------------------------------------
// LogLevel
public enum LogLevel: Int, Comparable {
    case error, warning, status, diagnostic

    @inlinable
    public init?(string: String) {
        switch string {
        case "error"     : self = .error
        case "warning"   : self = .warning
        case "status"    : self = .status
        case "diagnostic": self = .diagnostic
        default: return nil
        }
    }
    
    public static let maxStringWidth =
        String(describing: LogLevel.diagnostic).count
}

@inlinable
public func<(lhs: LogLevel, rhs: LogLevel) -> Bool {
    lhs.rawValue < rhs.rawValue
}
