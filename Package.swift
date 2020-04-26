// swift-tools-version:5.2
// The swift-tools-version declares the minimum version
// of Swift required to build this package.
import PackageDescription

let exclusions: [String] = ["platform/cuda", "*.gyb"]

let package = Package(
    name: "SwiftRT",
    products: [
        .library(name: "SwiftRT", targets: ["SwiftRT"]),
        .library(name: "SwiftRTLayers", targets: ["SwiftRTLayers"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", .branch("master"))
    ],
    targets: [
        .target(name: "SwiftRT",
                dependencies: [.product(name: "Numerics", package: "swift-numerics")],
                exclude: exclusions),
        
        .target(name: "SwiftRTLayers",
                dependencies: ["SwiftRT"],
                exclude: exclusions),
        
        .testTarget(name: "SwiftRTTests", dependencies: ["SwiftRT"]),
        .testTarget(name: "SwiftRTLayerTests", dependencies: ["SwiftRTLayers"]),
    ]
)
