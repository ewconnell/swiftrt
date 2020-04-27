// swift-tools-version:5.2
// The swift-tools-version declares the minimum version
// of Swift required to build this package.
import PackageDescription

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
                exclude: ["platform/cuda", "*.gyb"]),

        // tests
        .testTarget(name: "SwiftRTCoreTests", dependencies: ["SwiftRT"]),
        .testTarget(name: "SwiftRTLayerTests", dependencies: ["SwiftRT"]),
    ]
)
