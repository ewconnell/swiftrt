import XCTest
import SwiftRT

// TODO: Convert message to an @autoclosure.

func assertEqual<T: FloatingPoint>(
    _ x: [T], _ y: [T], accuracy: T, _ message: String = "",
    file: StaticString = #file, line: UInt = #line)
{
    for (x, y) in zip(x, y) {
        if x.isNaN || y.isNaN {
            XCTAssertTrue(x.isNaN && y.isNaN,
                          "\(x) is not equal to \(y) - \(message)",
                          file: file, line: line)
        } else {
            XCTAssertEqual(x, y, accuracy: accuracy, message,
                           file: file, line: line)
        }
    }
}

func assertEqual<Shape, T: FloatingPoint>(
    _ x: Tensor<Shape, T>,
    _ y: Tensor<Shape, T>,
    accuracy: T,
    _ message: String = "",
    file: StaticString = #file,
    line: UInt = #line
) where T.Value == T {
    assertEqual(x.flatArray, y.flatArray, accuracy: accuracy,
                message, file: file, line: line)
}
