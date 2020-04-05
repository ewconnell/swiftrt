// swift-tools-version:5.1
// The swift-tools-version declares the minimum version
// of Swift required to build this package.
import PackageDescription
import Foundation

#if os(Linux)
import Glibc
#else
import Darwin.C
#endif

//==============================================================================
// determine platform build type
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
]
var dependencies: [Target.Dependency] = ["Numerics"]
var testDependencies: [Target.Dependency] = ["SwiftRT"]
var exclusions: [String] = ["*.gyb"]
var targets: [PackageDescription.Target] = []

//==============================================================================
// Cuda service module
if buildCuda {
    //---------------------------------------
    // add Cuda system module
    products.append(.library(name: "CCuda", targets: ["CCuda"]))
    dependencies.append("CCuda")
    testDependencies.append("CCuda")

    #if os(Linux)
    targets.append(.systemLibrary(name: "CCuda", path: "Modules/Cuda",
            pkgConfig: "cuda"))
    #else
    targets.append(.systemLibrary(name: "CCuda", path: "Modules/Cuda",
            pkgConfig: "cuda_mac"))
    #endif
    
    //---------------------------------------
    // add SwiftRT Cuda kernels library built first via cmake
//    products.append(.library(name: "CudaKernels", targets: ["CudaKernels"]))
//    dependencies.append("CudaKernels")
//    testDependencies.append("CudaKernels")
//
//    targets.append(.systemLibrary(name: "CudaKernels", path: "Modules/CudaKernels"))

} else {
    exclusions.append("platform/cuda")
}

//==============================================================================
// package specification
targets.append(
    .target(name: "SwiftRT", dependencies: dependencies, exclude: exclusions))

targets.append(
    .testTarget(name: "SwiftRTTests", dependencies: testDependencies))

let package = Package(
    name: "SwiftRT",
    platforms: [
        .macOS(.v10_15),
    ],
    products: products,
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", .branch("master"))
    ],
    targets: targets
)
