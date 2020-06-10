// swift-tools-version:5.2
// The swift-tools-version declares the minimum version
// of Swift required to build this package.
import PackageDescription
import Foundation

// platform configuration
//let validPlatforms = Set(arrayLiteral: "cpu", "cuda")
//let environment = ProcessInfo.processInfo.environment
//let platform = (environment["SWIFTRT_PLATFORM"] ?? "cpu").lowercased()
//if !validPlatforms.contains(platform) {
//    fatalError("valid SWIFTRT_PLATFORM types: \(validPlatforms)")
//}
//let buildCuda = platform == "cuda"

#if os(Linux)
let exclusions = ["*.gyb"]
#else
let exclusions = ["*.gyb", "platform/cuda"]
#endif

// package definition
let package = Package(
    name: "SwiftRT",
    products: [
        .library(name: "SwiftRT", targets: ["SwiftRT"]),
        .library(name: "SwiftRTCore", targets: ["SwiftRTCore"]),
        .library(name: "SwiftRTLayers", targets: ["SwiftRTLayers"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", .branch("master"))
    ],
    targets: [
        // umbrella import
        .target(name: "SwiftRT", dependencies: ["SwiftRTCore", "SwiftRTLayers"]),
        
        // neural net layers
        .target(name: "SwiftRTLayers", dependencies: ["SwiftRTCore"]),
        
        // core platform and base types
        .target(name: "SwiftRTCore",
                dependencies: [.product(name: "Numerics", package: "swift-numerics")],
                exclude: exclusions),

        // tests
        .testTarget(name: "SwiftRTCoreTests", dependencies: ["SwiftRT"]),
        .testTarget(name: "SwiftRTLayerTests", dependencies: ["SwiftRT"]),
    ]
)
