// swift-tools-version:5.2
// The swift-tools-version declares the minimum version
// of Swift required to build this package.
import PackageDescription
import Foundation

//------------------------------------------------------------------------------
// determine platform build type "cpu" or "cuda"
let validPlatforms = Set(arrayLiteral: "cpu", "cuda")
let environment = ProcessInfo.processInfo.environment
let platform = (environment["SWIFTRT_PLATFORM"] ?? "cpu").lowercased()
if !validPlatforms.contains(platform) {
    fatalError("valid SWIFTRT_PLATFORM types: \(validPlatforms)")
}
let buildCuda = platform == "cuda"

//---------------------------------------
// the base products, dependencies, and targets
var products: [PackageDescription.Product] = [
    .library(name: "SwiftRT", targets: ["SwiftRT"]),
    .library(name: "SwiftRTCore", targets: ["SwiftRTCore"]),
    .library(name: "SwiftRTLayers", targets: ["SwiftRTLayers"]),
]

var targets: [PackageDescription.Target] = []
var coreDependencies: [Target.Dependency] =
        [.product(name: "Numerics", package: "swift-numerics")]

var testDependencies: [Target.Dependency] = ["SwiftRT"]
var exclusions: [String] = ["*.gyb"]

//==============================================================================
// Cuda platform module
if buildCuda {
    //---------------------------------------
    // add Cuda system module
    products.append(.library(name: "CCuda", targets: ["CCuda"]))
    coreDependencies.append("CCuda")
    testDependencies.append("CCuda")
    
    targets.append(
        .systemLibrary(name: "CCuda", path: "Modules/Cuda", pkgConfig: "cuda"))
} else {
    exclusions.append("platform/cuda")
}

//==============================================================================
// Targets
targets.append(contentsOf: [
    // umbrella import
    .target(name: "SwiftRT", dependencies: ["SwiftRTCore", "SwiftRTLayers"]),
    
    // neural net layers
    .target(name: "SwiftRTLayers", dependencies: ["SwiftRTCore"]),
    
    // core platform and base types
    .target(name: "SwiftRTCore", dependencies: coreDependencies, exclude: exclusions),
    
    // tests
    .testTarget(name: "SwiftRTCoreTests", dependencies: testDependencies),
    .testTarget(name: "SwiftRTLayerTests", dependencies: testDependencies),
])

//==============================================================================
// package specification
let package = Package(
    name: "SwiftRT",
    products: products,
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", .branch("master"))
    ],
    targets: targets
)
