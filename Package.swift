// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of 
// Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftRT",
    products: [
        .library(name: "SwiftRT", targets: ["SwiftRT"]),
        .library(name: "SwiftRTCore", targets: ["SwiftRTCore"]),
        .library(name: "SwiftRTLayers", targets: ["SwiftRTLayers"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics",
        .branch("main"))
    ],
    targets: [
        // umbrella import
        .target(
            name: "SwiftRT",
            dependencies: ["SwiftRTCore", "SwiftRTLayers"],
            exclude: ["CMakeLists.txt", "README.md"]),
        
        // neural net layers
        .target(
            name: "SwiftRTLayers",
            dependencies: ["SwiftRTCore"],
            exclude: ["CMakeLists.txt"]),
        
        // core platform and base types
        .target(
            name: "SwiftRTCore",
            dependencies: [.product(name: "Numerics", package: "swift-numerics")],
            exclude: [
                "CMakeLists.txt",
                "platform/cuda",
                "numpy/RankFunctions.swift.gyb",
                "numpy/array.swift.gyb",
                "numpy/arrayInit.swift.gyb",
                "tensor/Subscripts.swift.gyb",
            ]),
        
        // tests
        .testTarget(
            name: "BenchmarkTests", 
            dependencies: ["SwiftRT"],
            exclude: ["CMakeLists.txt"]),

        .testTarget(
            name: "SwiftRTCoreTests",
            dependencies: ["SwiftRT"],
            exclude: ["CMakeLists.txt"]),

        .testTarget(
            name: "SwiftRTLayerTests",
            dependencies: ["SwiftRT"],
            exclude: ["CMakeLists.txt"]),
    ],
    cxxLanguageStandard: .cxx1z
)
